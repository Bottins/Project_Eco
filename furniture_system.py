"""
Sistema completo di riconoscimento mobili con fine-tuning YOLOv8
- Fine-tuning automatico su dataset di mobili
- Classificazione stato oggetti (buono, da aggiustare, da buttare)
- Caching dei modelli addestrati
- Calcolo volume basato su dizionario
- GUI con visualizzazione multi-camion progressiva
- MODIFICHE: Filtro solo oggetti con misure + salvataggio crop in Output_Dataset
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import yaml
import shutil
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageTk
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import warnings
warnings.filterwarnings('ignore')

# Se esiste una root precedente, distruggila
try:
    if tk._default_root is not None:
        tk._default_root.destroy()
        tk._default_root = None
except Exception:
    pass

# Setup encoding per caratteri speciali
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass


class FurnitureDatasetPreparer:
    """Prepara il dataset per il fine-tuning di YOLO"""
    
    def __init__(self, data_dir: str = "dataset_mobili"):
        self.data_dir = Path(data_dir)
        self.classes = [
            'bed', 'couch', 'chair', 'dining_table', 'desk', 
            'wardrobe', 'bookshelf', 'tv_stand', 'coffee_table',
            'mattress', 'refrigerator', 'washing_machine', 'dishwasher'
        ]
        
    def prepare_yolo_dataset(self, images_dir: str, labels_dir: str):
        """
        Prepara il dataset nel formato YOLO
        Struttura attesa:
        - images_dir/: contiene le immagini (.jpg, .png)
        - labels_dir/: contiene le annotazioni (.txt) in formato YOLO
        """
        dataset_path = self.data_dir / "yolo_dataset"
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Crea struttura YOLO
        for split in ['train', 'val']:
            (dataset_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (dataset_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Dividi dataset
        images = list(Path(images_dir).glob("*.jpg")) + list(Path(images_dir).glob("*.png"))
        labels = list(Path(labels_dir).glob("*.txt"))
        
        if not images:
            print(f"[!] Nessuna immagine trovata in {images_dir}")
            return None
            
        # Split 80-20
        train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)
        
        # Copia file
        for img_list, split in [(train_imgs, 'train'), (val_imgs, 'val')]:
            for img_path in img_list:
                # Copia immagine
                dest_img = dataset_path / split / 'images' / img_path.name
                shutil.copy2(img_path, dest_img)
                
                # Copia label corrispondente
                label_path = Path(labels_dir) / f"{img_path.stem}.txt"
                if label_path.exists():
                    dest_label = dataset_path / split / 'labels' / f"{img_path.stem}.txt"
                    shutil.copy2(label_path, dest_label)
        
        # Crea file yaml per YOLO
        yaml_content = {
            'path': str(dataset_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'names': {i: cls for i, cls in enumerate(self.classes)}
        }
        
        yaml_path = dataset_path / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f)
        
        print(f"[OK] Dataset preparato in {dataset_path}")
        print(f"   Train: {len(train_imgs)} immagini")
        print(f"   Val: {len(val_imgs)} immagini")
        
        return yaml_path


class QualityClassifier(nn.Module):
    """Rete neurale per classificare lo stato degli oggetti"""
    
    def __init__(self, num_classes=3):
        super(QualityClassifier, self).__init__()
        # Usa ResNet18 pre-trained come backbone
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        
        # Sostituisci l'ultimo layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)


class QualityDataset(Dataset):
    """Dataset per training classificatore qualita"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class FurnitureVisualizerGUI:
    """GUI per visualizzare le detection in tempo reale con indicatore multi-camion"""
    
    def __init__(self, capacita_camion_m3: float = 4.0, detector_system=None):
        self.capacita_camion = capacita_camion_m3
        self.detections_list = []
        self.current_index = 0
        self.detector_system = detector_system
        
        # Setup finestra principale
        self.root = tk.Tk()
        self.root.title("Sistema Riconoscimento Mobili - Visualizzatore")
        
        # ADATTAMENTO SCHERMO
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Usa 85% dello schermo, max 1400x900
        window_width = min(1400, int(screen_width * 0.85))
        window_height = min(900, int(screen_height * 0.85))
        
        # Centra la finestra
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.minsize(800, 600)  # Dimensione minima
        self.root.configure(bg='#2b2b2b')
        
        # Frame principale
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame superiore per immagine
        image_frame = tk.Frame(main_frame, bg='#1e1e1e', relief=tk.RIDGE, bd=2)
        image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Canvas per immagine
        self.image_label = tk.Label(image_frame, bg='#1e1e1e')
        self.image_label.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        # Frame inferiore per info
        bottom_frame = tk.Frame(main_frame, bg='#2b2b2b')
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Dividi in due colonne: 1/3 sinistra, 2/3 destra
        bottom_frame.columnconfigure(0, weight=1)  # 1/3 per left_info
        bottom_frame.columnconfigure(1, weight=2)  # 2/3 per right_info
        
        left_info = tk.Frame(bottom_frame, bg='#2b2b2b')
        left_info.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        
        right_info = tk.Frame(bottom_frame, bg='#2b2b2b')
        right_info.grid(row=0, column=1, sticky='nsew')
        
        # Info oggetto corrente (sinistra)
        info_header = tk.Label(left_info, text="OGGETTO RILEVATO", 
                              font=("Arial", 14, "bold"), bg='#2b2b2b', fg='#00ff00')
        info_header.pack(anchor='w', pady=(0, 5))
        
        self.info_text = tk.Text(left_info, height=6, width=30, 
                                font=("Courier", 11), bg='#1e1e1e', fg='#ffffff',
                                relief=tk.FLAT, padx=10, pady=10)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # Camion visualizer (destra)
        truck_header = tk.Label(right_info, text="CAPACITA' CAMION RIFIUTI", 
                               font=("Arial", 14, "bold"), bg='#2b2b2b', fg='#ff8800')
        truck_header.pack(anchor='w', pady=(0, 5))
        
        # Frame scrollabile per i camion
        truck_container = tk.Frame(right_info, bg='#1e1e1e', relief=tk.RIDGE, bd=2)
        truck_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas con scrollbar
        self.truck_canvas = tk.Canvas(truck_container, height=150, 
                                     bg='#1e1e1e', highlightthickness=0)
        scrollbar = tk.Scrollbar(truck_container, orient=tk.HORIZONTAL, command=self.truck_canvas.xview)
        self.truck_canvas.configure(xscrollcommand=scrollbar.set)
        
        self.truck_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Label percentuale
        self.percent_label = tk.Label(right_info, text="0.00 m³ accumulati (0 camion)", 
                                     font=("Arial", 12, "bold"), bg='#2b2b2b', fg='#ffffff')
        self.percent_label.pack(pady=5)
        
        # Controlli navigazione
        controls_frame = tk.Frame(main_frame, bg='#2b2b2b')
        controls_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
        
        # Bottoni
        btn_style = {'font': ("Arial", 11, "bold"), 'bg': '#0066cc', 'fg': 'white', 
                    'activebackground': '#0052a3', 'relief': tk.FLAT, 'padx': 20, 'pady': 8}
        
        self.prev_btn = tk.Button(controls_frame, text="Precedente", 
                                  command=self.prev_detection, **btn_style)
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        
        self.counter_label = tk.Label(controls_frame, text="0 / 0", 
                                     font=("Arial", 12, "bold"), bg='#2b2b2b', fg='#ffffff')
        self.counter_label.pack(side=tk.LEFT, padx=20)
        
        self.next_btn = tk.Button(controls_frame, text="Successivo",
                                 command=self.next_detection, **btn_style)
        self.next_btn.pack(side=tk.LEFT, padx=5)

        # Pulsante Verifica Ritiro
        verify_btn_style = {'font': ("Arial", 11, "bold"), 'bg': '#ff8800', 'fg': 'white',
                           'activebackground': '#cc6600', 'relief': tk.FLAT, 'padx': 20, 'pady': 8}
        self.verify_btn = tk.Button(controls_frame, text="📸 Verifica Ritiro",
                                    command=self.verifica_ritiro, **verify_btn_style)
        self.verify_btn.pack(side=tk.LEFT, padx=20)

        # Abilita frecce tastiera
        self.root.bind("<Left>", lambda e: self.prev_detection())
        self.root.bind("<Right>", lambda e: self.next_detection())

        self.close_btn = tk.Button(controls_frame, text="Chiudi",
                                   command=self.close, bg='#cc0000', fg='white',
                                   activebackground='#a30000', relief=tk.FLAT,
                                   font=("Arial", 11, "bold"), padx=20, pady=8)
        self.close_btn.pack(side=tk.RIGHT, padx=5)
    
    def calculate_accumulated_volume(self) -> float:
        """Calcola il volume totale accumulato fino all'indice corrente"""
        volume_totale = 0.0
        for i in range(self.current_index + 1):
            det = self.detections_list[i]
            volume = det['volume_stimato'].get('volume_m3', 0) or 0
            volume_totale += volume
        return volume_totale
    
    def draw_trucks(self, volume_totale: float):
        """Disegna multipli camion in base al volume totale"""
        self.truck_canvas.delete("all")
        
        # Calcola quanti camion servono
        num_camion_pieni = int(volume_totale // self.capacita_camion)
        volume_parziale = volume_totale % self.capacita_camion
        percentuale_parziale = (volume_parziale / self.capacita_camion) * 100
        
        # Se c'è volume parziale, serve un camion in più
        num_camion_totali = num_camion_pieni + (1 if volume_parziale > 0 else 0)
        
        # Dimensioni singolo camion
        truck_width = 160
        truck_height = 70
        spacing = 70
        y_start = 20
        
        # Calcola larghezza totale necessaria
        total_width = num_camion_totali * (truck_width + spacing) + spacing
        self.truck_canvas.configure(scrollregion=(0, 0, total_width, 250))
        
        # Disegna ogni camion
        for i in range(num_camion_totali):
            x_start = spacing + i * (truck_width + spacing)
            
            # Determina la percentuale di riempimento
            if i < num_camion_pieni:
                fill_percentage = 100  # Camion pieno
            else:
                fill_percentage = percentuale_parziale  # Camion parziale
            
            self.draw_single_truck(x_start, y_start, truck_width, truck_height, 
                                  fill_percentage, i + 1)
    
    def draw_single_truck(self, x_start, y_start, truck_width, truck_height, 
                         fill_percentage, truck_number):
        """Disegna un singolo camion con indicatore di riempimento"""
        
        # Calcola altezza riempimento
        fill_height = int((truck_height - 4) * (fill_percentage / 100))
        
        # Disegna container del camion (bordo)
        self.truck_canvas.create_rectangle(
            x_start, y_start, x_start + truck_width, y_start + truck_height,
            outline='#888888', width=3, fill='#333333'
        )
        
        # Parte riempita (rosso dal basso)
        if fill_height > 0:
            self.truck_canvas.create_rectangle(
                x_start + 2, y_start + truck_height - fill_height - 2,
                x_start + truck_width - 2, y_start + truck_height - 2,
                fill='#ff3333', outline=''
            )
        
        # Parte vuota (verde dall'alto)
        empty_height = truck_height - fill_height - 4
        if empty_height > 0:
            self.truck_canvas.create_rectangle(
                x_start + 2, y_start + 2,
                x_start + truck_width - 2, y_start + 2 + empty_height,
                fill='#33ff33', outline=''
            )
        
        # Cabina del camion
        cabin_width = 55
        cabin_height = 75
        cabin_x = x_start + truck_width
        cabin_y = y_start + truck_height - cabin_height
        
        self.truck_canvas.create_rectangle(
            cabin_x, cabin_y, cabin_x + cabin_width, cabin_y + cabin_height,
            fill='#ff8800', outline='#cc6600', width=2
        )
        
        # Finestra cabina
        window_margin = 8
        self.truck_canvas.create_rectangle(
            cabin_x + window_margin, cabin_y + window_margin,
            cabin_x + cabin_width - window_margin, cabin_y + 25,
            fill='#87ceeb', outline='#5599cc', width=2
        )
        
        # Ruote
        wheel_radius = 14
        wheel_y = y_start + truck_height + wheel_radius - 5
        
        # Ruote container
        wheel1_x = x_start + 35
        wheel2_x = x_start + truck_width - 35
        
        for wheel_x in [wheel1_x, wheel2_x]:
            self.truck_canvas.create_oval(
                wheel_x - wheel_radius, wheel_y - wheel_radius,
                wheel_x + wheel_radius, wheel_y + wheel_radius,
                fill='#222222', outline='#888888', width=2
            )
            # Cerchione
            self.truck_canvas.create_oval(
                wheel_x - wheel_radius//2, wheel_y - wheel_radius//2,
                wheel_x + wheel_radius//2, wheel_y + wheel_radius//2,
                fill='#555555', outline=''
            )
        
        # Ruota cabina
        wheel3_x = cabin_x + cabin_width//2
        self.truck_canvas.create_oval(
            wheel3_x - wheel_radius, wheel_y - wheel_radius,
            wheel3_x + wheel_radius, wheel_y + wheel_radius,
            fill='#222222', outline='#888888', width=2
        )
        self.truck_canvas.create_oval(
            wheel3_x - wheel_radius//2, wheel_y - wheel_radius//2,
            wheel3_x + wheel_radius//2, wheel_y + wheel_radius//2,
            fill='#555555', outline=''
        )
        
        # Linee di divisione container
        for i in range(1, 4):
            y_line = y_start + (truck_height // 4) * i
            self.truck_canvas.create_line(
                x_start, y_line, x_start + truck_width, y_line,
                fill='#555555', width=1, dash=(5, 5)
            )
        
        # Testo percentuale sul camion
        text_y = y_start + truck_height // 2
        self.truck_canvas.create_text(
            x_start + truck_width // 2, text_y,
            text=f"{fill_percentage:.1f}%",
            font=("Arial", 20, "bold"), fill='#ffffff'
        )
        
        # Numero camion sotto
        self.truck_canvas.create_text(
            x_start + truck_width // 2 + cabin_width // 2, y_start + truck_height + 35,
            text=f"Camion #{truck_number}",
            font=("Arial", 10, "bold"), fill='#aaaaaa'
        )
    
    def update_display(self):
        """Aggiorna la visualizzazione con la detection corrente"""
        if not self.detections_list or self.current_index >= len(self.detections_list):
            return
        
        det = self.detections_list[self.current_index]
        
        # Calcola volume accumulato
        volume_accumulato = self.calculate_accumulated_volume()
        
        # Carica e mostra immagine
        img_path = det['immagine']
        image = cv2.imread(str(img_path))
        if image is None:
            return
        
        # Disegna solo il bbox dell'oggetto corrente
        annotated = image.copy()
        bbox = det['bbox']
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        
        # Colore in base alla qualità
        colors = {
            'buono': (0, 255, 0),
            'da_aggiustare': (0, 165, 255),
            'da_buttare': (0, 0, 255),
            'non_valutabile': (128, 128, 128)
        }
        qualita = det['qualita']['categoria']
        color = colors.get(qualita, (255, 255, 255))
        
        # Disegna bbox più spesso
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 4)
        
        # Ridimensiona per finestra
        # Ridimensiona per finestra (adattivo)
        window_width = self.root.winfo_width()
        window_height = self.root.winfo_height()
        max_width = min(900, int(window_width * 0.65)) if window_width > 100 else 900
        max_height = min(500, int(window_height * 0.45)) if window_height > 100 else 500
        h, w = annotated.shape[:2]
        scale = min(max_width/w, max_height/h)
        new_w, new_h = int(w*scale), int(h*scale)
        
        annotated_resized = cv2.resize(annotated, (new_w, new_h))
        
        # Converti per Tkinter
        annotated_rgb = cv2.cvtColor(annotated_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(annotated_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk
        
        # Aggiorna info testuale
        self.info_text.delete(1.0, tk.END)
        
        volume = det['volume_stimato'].get('volume_m3', 0) or 0
        
        info_lines = [
            f"RIFIUTO INDIVIDUATO",
            f"━━━━━━━━━━━━━━━━━━━━━",
            f"",
            f"Tipo:        {det['classe'].upper()}",
            f"Sottotipo:   {det['sottotipo']}",
            f"",
            f"Stato:       {qualita.upper().replace('_', ' ')}",
            f"Affidabilita: {det['qualita']['confidence']*100:.1f}%",
            f"",
            f"Volume:      {volume:.3f} m³",
            f"",
            f"━━━━━━━━━━━━━━━━━━━━━",
            f"Volume accumulato: {volume_accumulato:.3f} m³"
        ]
        
        self.info_text.insert(1.0, '\n'.join(info_lines))
        
        # Aggiorna camion
        self.draw_trucks(volume_accumulato)
        
        # Calcola numero camion necessari
        num_camion_pieni = int(volume_accumulato // self.capacita_camion)
        volume_parziale = volume_accumulato % self.capacita_camion
        num_camion_totali = num_camion_pieni + (1 if volume_parziale > 0 else 0)
        
        # Aggiorna label
        if num_camion_totali == 0:
            label_text = f"{volume_accumulato:.3f} m³ accumulati (0 camion)"
        elif num_camion_totali == 1:
            percentuale = (volume_accumulato / self.capacita_camion) * 100
            label_text = f"{volume_accumulato:.3f} m³ accumulati (1 camion al {percentuale:.1f}%)"
        else:
            if volume_parziale > 0:
                percentuale_ultimo = (volume_parziale / self.capacita_camion) * 100
                label_text = f"{volume_accumulato:.3f} m³ accumulati ({num_camion_pieni} camion pieni + 1 al {percentuale_ultimo:.1f}%)"
            else:
                label_text = f"{volume_accumulato:.3f} m³ accumulati ({num_camion_pieni} camion pieni)"
        
        self.percent_label.configure(text=label_text)
        
        # Cambia colore in base al riempimento dell'ultimo camion
        percentuale_ultimo = (volume_parziale / self.capacita_camion) * 100 if volume_parziale > 0 else 100
        if percentuale_ultimo >= 100:
            self.percent_label.configure(fg='#ff3333')
        elif percentuale_ultimo >= 80:
            self.percent_label.configure(fg='#ff8800')
        else:
            self.percent_label.configure(fg='#ffffff')
        
        # Aggiorna contatore
        self.counter_label.configure(
            text=f"{self.current_index + 1} / {len(self.detections_list)}"
        )
        
        # Abilita/disabilita bottoni
        self.prev_btn.configure(state=tk.NORMAL)
        self.next_btn.configure(state=tk.NORMAL)
    
    def next_detection(self):
        """Passa alla detection successiva"""
        if self.current_index < len(self.detections_list) - 1:
            self.current_index += 1
            self.update_display()
    
    def prev_detection(self):
        """Torna alla detection precedente"""
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()
    
    def close(self):
        """Chiude la finestra"""
        self.root.quit()
        self.root.destroy()

    def verifica_ritiro(self):
        """Apre dialog per caricare foto di verifica e confronta con originale"""
        if not self.detector_system:
            messagebox.showerror("Errore", "Sistema di detection non disponibile")
            return

        if not self.detections_list or self.current_index >= len(self.detections_list):
            messagebox.showerror("Errore", "Nessun oggetto selezionato")
            return

        # Apri dialog per selezionare immagine
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(
            title="Seleziona foto di verifica ritiro",
            filetypes=[
                ("Immagini", "*.jpg *.jpeg *.png *.bmp"),
                ("Tutti i file", "*.*")
            ]
        )

        if not file_path:
            return  # Utente ha annullato

        # Mostra dialog di caricamento
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Analisi in corso...")
        progress_window.geometry("300x100")
        progress_window.transient(self.root)
        progress_window.grab_set()

        # Centra la finestra
        progress_window.update_idletasks()
        x = (progress_window.winfo_screenwidth() // 2) - (progress_window.winfo_width() // 2)
        y = (progress_window.winfo_screenheight() // 2) - (progress_window.winfo_height() // 2)
        progress_window.geometry(f"+{x}+{y}")

        label = tk.Label(progress_window, text="Analisi della foto in corso...\nAttendere prego.",
                        font=("Arial", 11), pady=20)
        label.pack()

        # Esegui detection in modo asincrono
        def esegui_detection():
            try:
                # Rileva oggetti nella nuova foto
                nuove_detection = self.detector_system.rileva_mobili(file_path)

                # Ottieni detection originale per l'immagine corrente
                det_corrente = self.detections_list[self.current_index]
                img_originale = det_corrente['immagine']

                # Rileva oggetti nell'immagine originale (tutte le detection)
                detection_originali = [d for d in self.detections_list if d['immagine'] == img_originale]

                # Confronta
                risultato = self.confronta_oggetti(detection_originali, nuove_detection)

                # Chiudi progress
                progress_window.destroy()

                # Mostra risultati
                self.mostra_risultati_confronto(risultato, file_path, img_originale)

            except Exception as e:
                progress_window.destroy()
                messagebox.showerror("Errore", f"Errore durante l'analisi:\n{str(e)}")

        # Esegui dopo 100ms per permettere alla finestra di mostrarsi
        self.root.after(100, esegui_detection)

    def confronta_oggetti(self, originali: List[Dict], nuovi: List[Dict]) -> Dict:
        """Confronta oggetti rilevati e identifica quelli aggiunti"""

        # Conta oggetti per classe nell'originale
        classi_originali = {}
        for det in originali:
            classe = det['classe']
            classi_originali[classe] = classi_originali.get(classe, 0) + 1

        # Conta oggetti per classe nella nuova foto
        classi_nuove = {}
        for det in nuovi:
            classe = det['classe']
            classi_nuove[classe] = classi_nuove.get(classe, 0) + 1

        # Identifica oggetti aggiunti
        oggetti_aggiunti = []
        volume_aggiunto = 0.0

        for classe, count_nuovo in classi_nuove.items():
            count_originale = classi_originali.get(classe, 0)
            if count_nuovo > count_originale:
                diff = count_nuovo - count_originale
                # Trova gli oggetti di questa classe nella nuova detection
                oggetti_classe = [d for d in nuovi if d['classe'] == classe]
                # Prendi gli ultimi 'diff' oggetti (quelli aggiunti)
                for obj in oggetti_classe[-diff:]:
                    oggetti_aggiunti.append(obj)
                    volume_aggiunto += obj['volume_stimato'].get('volume_m3', 0) or 0

        # Identifica oggetti rimossi
        oggetti_rimossi = []
        for classe, count_originale in classi_originali.items():
            count_nuovo = classi_nuove.get(classe, 0)
            if count_nuovo < count_originale:
                diff = count_originale - count_nuovo
                oggetti_rimossi.append({'classe': classe, 'quantita': diff})

        return {
            'totale_originali': len(originali),
            'totale_nuovi': len(nuovi),
            'classi_originali': classi_originali,
            'classi_nuove': classi_nuove,
            'oggetti_aggiunti': oggetti_aggiunti,
            'oggetti_rimossi': oggetti_rimossi,
            'volume_aggiunto': volume_aggiunto,
            'nuove_detection': nuovi
        }

    def mostra_risultati_confronto(self, risultato: Dict, img_nuova: str, img_originale: str):
        """Mostra finestra con i risultati del confronto"""

        # Crea finestra di risultati
        result_window = tk.Toplevel(self.root)
        result_window.title("Risultati Verifica Ritiro")
        result_window.geometry("900x700")
        result_window.configure(bg='#2b2b2b')

        # Centra la finestra
        result_window.update_idletasks()
        x = (result_window.winfo_screenwidth() // 2) - (result_window.winfo_width() // 2)
        y = (result_window.winfo_screenheight() // 2) - (result_window.winfo_height() // 2)
        result_window.geometry(f"+{x}+{y}")

        # Frame principale
        main_frame = tk.Frame(result_window, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Titolo
        num_aggiunti = len(risultato['oggetti_aggiunti'])
        if num_aggiunti > 0:
            title_text = f"⚠️ ATTENZIONE: {num_aggiunti} OGGETTO/I AGGIUNTO/I!"
            title_color = '#ff3333'
        else:
            title_text = "✓ NESSUN OGGETTO AGGIUNTO"
            title_color = '#00ff00'

        title = tk.Label(main_frame, text=title_text, font=("Arial", 16, "bold"),
                        bg='#2b2b2b', fg=title_color)
        title.pack(pady=(0, 20))

        # Frame per immagini (side by side)
        images_frame = tk.Frame(main_frame, bg='#2b2b2b')
        images_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))

        # Immagine originale
        left_frame = tk.Frame(images_frame, bg='#1e1e1e', relief=tk.RIDGE, bd=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        left_label = tk.Label(left_frame, text="FOTO ORIGINALE", font=("Arial", 12, "bold"),
                             bg='#1e1e1e', fg='#ffffff')
        left_label.pack(pady=5)

        left_img_label = tk.Label(left_frame, bg='#1e1e1e')
        left_img_label.pack(expand=True, padx=5, pady=5)

        # Immagine nuova
        right_frame = tk.Frame(images_frame, bg='#1e1e1e', relief=tk.RIDGE, bd=2)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_label = tk.Label(right_frame, text="FOTO VERIFICA", font=("Arial", 12, "bold"),
                              bg='#1e1e1e', fg='#ffffff')
        right_label.pack(pady=5)

        right_img_label = tk.Label(right_frame, bg='#1e1e1e')
        right_img_label.pack(expand=True, padx=5, pady=5)

        # Carica e mostra immagini
        def load_and_show_images():
            # Carica immagine originale
            img_orig = cv2.imread(str(img_originale))
            if img_orig is not None:
                # Disegna bbox su originale
                for det in self.detections_list:
                    if det['immagine'] == img_originale:
                        bbox = det['bbox']
                        cv2.rectangle(img_orig, (bbox['x1'], bbox['y1']),
                                    (bbox['x2'], bbox['y2']), (0, 255, 0), 3)

                # Ridimensiona
                h, w = img_orig.shape[:2]
                scale = min(400/w, 250/h)
                new_w, new_h = int(w*scale), int(h*scale)
                img_orig_resized = cv2.resize(img_orig, (new_w, new_h))
                img_orig_rgb = cv2.cvtColor(img_orig_resized, cv2.COLOR_BGR2RGB)
                img_orig_pil = Image.fromarray(img_orig_rgb)
                img_orig_tk = ImageTk.PhotoImage(img_orig_pil)
                left_img_label.configure(image=img_orig_tk)
                left_img_label.image = img_orig_tk

            # Carica immagine nuova
            img_new = cv2.imread(str(img_nuova))
            if img_new is not None:
                # Disegna bbox: verde per originali, rosso per aggiunti
                aggiunti_ids = set(id(obj) for obj in risultato['oggetti_aggiunti'])

                for det in risultato['nuove_detection']:
                    bbox = det['bbox']
                    # Rosso se aggiunto, verde altrimenti
                    color = (0, 0, 255) if id(det) in aggiunti_ids else (0, 255, 0)
                    thickness = 4 if id(det) in aggiunti_ids else 2
                    cv2.rectangle(img_new, (bbox['x1'], bbox['y1']),
                                (bbox['x2'], bbox['y2']), color, thickness)

                    # Aggiungi etichetta "NUOVO!" per oggetti aggiunti
                    if id(det) in aggiunti_ids:
                        cv2.putText(img_new, "NUOVO!", (bbox['x1'], bbox['y1']-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Ridimensiona
                h, w = img_new.shape[:2]
                scale = min(400/w, 250/h)
                new_w, new_h = int(w*scale), int(h*scale)
                img_new_resized = cv2.resize(img_new, (new_w, new_h))
                img_new_rgb = cv2.cvtColor(img_new_resized, cv2.COLOR_BGR2RGB)
                img_new_pil = Image.fromarray(img_new_rgb)
                img_new_tk = ImageTk.PhotoImage(img_new_pil)
                right_img_label.configure(image=img_new_tk)
                right_img_label.image = img_new_tk

        load_and_show_images()

        # Frame per dettagli
        details_frame = tk.Frame(main_frame, bg='#1e1e1e', relief=tk.RIDGE, bd=2)
        details_frame.pack(fill=tk.BOTH, expand=True)

        # Text widget per dettagli
        details_text = tk.Text(details_frame, height=12, font=("Courier", 10),
                              bg='#1e1e1e', fg='#ffffff', padx=15, pady=15)
        details_text.pack(fill=tk.BOTH, expand=True)

        # Compila dettagli
        details_lines = [
            "═══════════════════════════════════════════════════════════════════════",
            "                       RIEPILOGO CONFRONTO                             ",
            "═══════════════════════════════════════════════════════════════════════",
            "",
            f"Oggetti foto originale:  {risultato['totale_originali']}",
            f"Oggetti foto verifica:   {risultato['totale_nuovi']}",
            "",
            "OGGETTI ORIGINALI PER CLASSE:",
        ]

        for classe, count in risultato['classi_originali'].items():
            details_lines.append(f"  • {classe.upper()}: {count}")

        details_lines.append("")
        details_lines.append("OGGETTI RILEVATI NELLA VERIFICA:")

        for classe, count in risultato['classi_nuove'].items():
            details_lines.append(f"  • {classe.upper()}: {count}")

        if risultato['oggetti_aggiunti']:
            details_lines.append("")
            details_lines.append("⚠️  OGGETTI AGGIUNTI ILLEGALMENTE:")
            details_lines.append("")

            for obj in risultato['oggetti_aggiunti']:
                volume = obj['volume_stimato'].get('volume_m3', 0) or 0
                details_lines.append(f"  • {obj['classe'].upper()} ({obj['sottotipo']})")
                details_lines.append(f"    Volume: {volume:.3f} m³")
                details_lines.append(f"    Qualità: {obj['qualita']['categoria'].upper().replace('_', ' ')}")
                details_lines.append("")

            details_lines.append(f"VOLUME TOTALE AGGIUNTO: {risultato['volume_aggiunto']:.3f} m³")
        else:
            details_lines.append("")
            details_lines.append("✓ Nessun oggetto aggiunto rilevato")

        if risultato['oggetti_rimossi']:
            details_lines.append("")
            details_lines.append("ℹ️  OGGETTI RIMOSSI (Ritiro completato):")
            for obj in risultato['oggetti_rimossi']:
                details_lines.append(f"  • {obj['classe'].upper()}: -{obj['quantita']}")

        details_lines.append("")
        details_lines.append("═══════════════════════════════════════════════════════════════════════")

        details_text.insert(1.0, '\n'.join(details_lines))
        details_text.configure(state='disabled')

        # Bottone chiudi
        close_btn = tk.Button(main_frame, text="Chiudi", command=result_window.destroy,
                             bg='#0066cc', fg='white', font=("Arial", 11, "bold"),
                             padx=30, pady=10, relief=tk.FLAT)
        close_btn.pack(pady=(10, 0))

    def show(self, detections: List[Dict]):
        """Mostra la GUI con le detection"""
        if not detections:
            print("[INFO] Nessuna detection da visualizzare")
            return
        
        self.detections_list = detections
        self.current_index = 0
        
        self.root.after(50, self.update_display)
        self.root.mainloop()


class FurnitureDetectorSystem:
    """Sistema principale per detection e classificazione mobili"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Path dei modelli
        self.detector_path = self.models_dir / "furniture_detector.pt"
        self.quality_path = self.models_dir / "quality_classifier.pth"
        
        # Dizionario dimensioni (in cm)
        self.dimensioni_standard = {
            'bed': {
                'singolo': {'lunghezza': 200, 'larghezza': 90, 'altezza': 50},
                'matrimoniale': {'lunghezza': 200, 'larghezza': 160, 'altezza': 50},
                'king': {'lunghezza': 200, 'larghezza': 180, 'altezza': 50}
            },
            'couch': {
                'piccolo': {'lunghezza': 150, 'profondita': 80, 'altezza': 85},
                'medio': {'lunghezza': 180, 'profondita': 85, 'altezza': 88},
                'grande': {'lunghezza': 220, 'profondita': 90, 'altezza': 90}
            },
            'chair': {
                'standard': {'lunghezza': 45, 'profondita': 50, 'altezza': 90},
                'poltrona': {'lunghezza': 70, 'profondita': 70, 'altezza': 100}
            },
            'dining_table': {
                'piccolo': {'lunghezza': 120, 'larghezza': 80, 'altezza': 75},
                'medio': {'lunghezza': 150, 'larghezza': 90, 'altezza': 75},
                'grande': {'lunghezza': 200, 'larghezza': 100, 'altezza': 75}
            },
            'desk': {
                'standard': {'lunghezza': 120, 'larghezza': 60, 'altezza': 75},
                'grande': {'lunghezza': 160, 'larghezza': 80, 'altezza': 75}
            },
            'wardrobe': {
                'piccolo': {'lunghezza': 100, 'profondita': 60, 'altezza': 200},
                'medio': {'lunghezza': 150, 'profondita': 60, 'altezza': 220},
                'grande': {'lunghezza': 200, 'profondita': 60, 'altezza': 240}
            },
            'bookshelf': {
                'standard': {'lunghezza': 80, 'profondita': 30, 'altezza': 180}
            },
            'refrigerator': {
                'standard': {'lunghezza': 60, 'profondita': 65, 'altezza': 170}
            },
            'washing_machine': {
                'standard': {'lunghezza': 60, 'profondita': 60, 'altezza': 85}
            }
        }
        
        # Classi qualita
        self.quality_classes = ['buono', 'da_aggiustare', 'da_buttare']
        
        # Carica o inizializza modelli
        self._initialize_models()
    
    def _initialize_models(self):
        """Inizializza o carica modelli esistenti"""
        # Detector YOLO
        if self.detector_path.exists():
            print(f"[OK] Caricamento detector esistente: {self.detector_path}")
            self.detector = YOLO(str(self.detector_path))
        else:
            print("[INFO] Download YOLOv8 base...")
            self.detector = YOLO('yolov8m.pt')
            print("[!] Detector non fine-tuned. Esegui train_furniture_detector() per il fine-tuning")
        
        # Quality classifier
        self.quality_model = QualityClassifier(num_classes=3)
        if self.quality_path.exists():
            print(f"[OK] Caricamento quality classifier esistente: {self.quality_path}")
            self.quality_model.load_state_dict(torch.load(self.quality_path, map_location='cpu'))
            self.quality_model.eval()
        else:
            print("[!] Quality classifier non addestrato. Esegui train_quality_classifier() per l'addestramento")
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.quality_model.to(self.device)
        
        # Transform per quality classifier
        self.quality_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def train_furniture_detector(self, dataset_yaml: str, epochs: int = 50, batch_size: int = 16):
        """Fine-tuning del detector YOLO per mobili"""
        print("\n" + "="*50)
        print("FINE-TUNING YOLO PER MOBILI")
        print("="*50)
        
        # Controlla se esiste gia
        if self.detector_path.exists():
            response = input("[!] Modello gia esistente. Vuoi riaddestrarlo? (s/n): ")
            if response.lower() != 's':
                print("Training annullato")
                return
        
        # Training
        print(f"\n[INFO] Avvio training per {epochs} epoche...")
        results = self.detector.train(
            data=dataset_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            device=0 if torch.cuda.is_available() else 'cpu',
            project=str(self.models_dir),
            name='furniture_training',
            exist_ok=True,
            patience=10,
            save=True,
            plots=True
        )
        
        # Salva il modello fine-tuned
        best_model_path = self.models_dir / 'furniture_training' / 'weights' / 'best.pt'
        if best_model_path.exists():
            shutil.copy2(best_model_path, self.detector_path)
            print(f"[OK] Modello salvato in: {self.detector_path}")
            
            # Ricarica il modello fine-tuned
            self.detector = YOLO(str(self.detector_path))
        else:
            print("[ERRORE] Errore nel salvataggio del modello")
    
    def calcola_iou(self, box1: Dict, box2: Dict) -> float:
        """
        Calcola IoU (Intersection over Union) tra due bounding box
        box format: {'x1': ..., 'y1': ..., 'x2': ..., 'y2': ...}
        """
        # Coordinate dell'intersezione
        x1_inter = max(box1['x1'], box2['x1'])
        y1_inter = max(box1['y1'], box2['y1'])
        x2_inter = min(box1['x2'], box2['x2'])
        y2_inter = min(box1['y2'], box2['y2'])
        
        # Area intersezione
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0  # Nessuna sovrapposizione
        
        area_inter = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Aree dei singoli box
        area_box1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
        area_box2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
        
        # IoU = intersezione / unione
        area_union = area_box1 + area_box2 - area_inter
        
        return area_inter / area_union if area_union > 0 else 0.0
    
    def filtra_box_sovrapposti(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """
        Rimuove detection sovrapposte, tenendo quella con area maggiore
        """
        if len(detections) <= 1:
            return detections
        
        # Ordina per area decrescente (più grandi prima)
        detections_sorted = sorted(
            detections, 
            key=lambda d: (d['bbox']['x2'] - d['bbox']['x1']) * (d['bbox']['y2'] - d['bbox']['y1']),
            reverse=True
        )
        
        detections_filtrate = []
        
        for det in detections_sorted:
            # Controlla se si sovrappone con detection già accettate
            sovrapposizione = False
            for det_accettata in detections_filtrate:
                iou = self.calcola_iou(det['bbox'], det_accettata['bbox'])
                if iou > iou_threshold:
                    sovrapposizione = True
                    break
            
            # Se non si sovrappone, accettala
            if not sovrapposizione:
                detections_filtrate.append(det)
        
        return detections_filtrate

    def train_quality_classifier(self, images_dir: str, labels_file: str, epochs: int = 30):
        """Training del classificatore di qualita"""
        print("\n" + "="*50)
        print("TRAINING CLASSIFICATORE QUALITA'")
        print("="*50)
        
        # Controlla se esiste gia
        if self.quality_path.exists():
            response = input("[!] Modello gia esistente. Vuoi riaddestrarlo? (s/n): ")
            if response.lower() != 's':
                print("Training annullato")
                return
        
        # Carica labels
        with open(labels_file, 'r') as f:
            labels_dict = json.load(f)
        
        # Prepara dataset
        image_paths = []
        labels = []
        
        for img_name, quality in labels_dict.items():
            img_path = Path(images_dir) / img_name
            if img_path.exists():
                image_paths.append(img_path)
                labels.append(self.quality_classes.index(quality))
        
        if not image_paths:
            print("[ERRORE] Nessuna immagine trovata per il training")
            return
        
        # Split dataset
        X_train, X_val, y_train, y_val = train_test_split(
            image_paths, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Data augmentation per training
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Datasets
        train_dataset = QualityDataset(X_train, y_train, train_transform)
        val_dataset = QualityDataset(X_val, y_val, self.quality_transform)
        
        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.quality_model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        # Training loop
        print(f"\n[INFO] Training su {len(X_train)} immagini...")
        best_val_acc = 0
        
        for epoch in range(epochs):
            # Training
            self.quality_model.train()
            train_loss = 0
            train_correct = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.quality_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_correct += predicted.eq(labels).sum().item()
            
            # Validation
            self.quality_model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.quality_model(images)
                    _, predicted = outputs.max(1)
                    val_correct += predicted.eq(labels).sum().item()
                    val_total += labels.size(0)
            
            val_acc = 100. * val_correct / val_total
            train_acc = 100. * train_correct / len(train_dataset)
            
            print(f"Epoch [{epoch+1}/{epochs}] - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")
            
            # Salva il miglior modello
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.quality_model.state_dict(), self.quality_path)
                print(f"  [SAVE] Modello salvato (Val Acc: {val_acc:.2f}%)")
            
            scheduler.step()
        
        print(f"\n[OK] Training completato! Miglior accuratezza: {best_val_acc:.2f}%")
    
    def calcola_volume(self, classe: str, bbox_width: float, bbox_height: float) -> Dict:
        """Calcola il volume stimato basandosi su dimensioni e classe"""
        if classe not in self.dimensioni_standard:
            return {'volume_m3': None, 'dimensioni_cm': None, 'sottotipo': 'unknown'}
        
        # Determina sottotipo basandosi sulle dimensioni del bbox
        sottotipi = self.dimensioni_standard[classe]
        
        # Logica semplificata per determinare il sottotipo
        if len(sottotipi) == 1:
            sottotipo = list(sottotipi.keys())[0]
        else:
            # Usa il rapporto larghezza/altezza per stimare
            if classe == 'bed':
                sottotipo = 'matrimoniale' if bbox_width > 300 else 'singolo'
            elif classe == 'couch':
                if bbox_width < 250:
                    sottotipo = 'piccolo'
                elif bbox_width < 350:
                    sottotipo = 'medio'
                else:
                    sottotipo = 'grande'
            else:
                # Default al primo sottotipo
                sottotipo = list(sottotipi.keys())[0]
        
        if sottotipo in sottotipi:
            dims = sottotipi[sottotipo]
            
            # Calcola volume in m³
            lunghezza = dims.get('lunghezza', 100)
            larghezza = dims.get('larghezza', dims.get('profondita', 100))
            altezza = dims.get('altezza', 100)
            
            volume = (lunghezza * larghezza * altezza) / 1_000_000
            
            return {
                'volume_m3': round(volume, 3),
                'dimensioni_cm': dims,
                'sottotipo': sottotipo
            }
        
        return {'volume_m3': None, 'dimensioni_cm': None, 'sottotipo': 'unknown'}
    
    def classifica_qualita(self, image: np.ndarray, bbox: List[int]) -> Dict:
        """Classifica la qualita di un oggetto ritagliato"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ritaglia l'oggetto
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                return {'categoria': 'non_valutabile', 'confidence': 0.0}
            
            # Converti per PIL
            roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            
            # Applica transform e predici
            img_tensor = self.quality_transform(roi_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.quality_model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)
            
            categoria = self.quality_classes[predicted.item()]
            
            return {
                'categoria': categoria,
                'confidence': float(confidence.item()),
                'probabilita': {
                    cls: float(probs[0][i].item()) 
                    for i, cls in enumerate(self.quality_classes)
                }
            }
        except Exception as e:
            print(f"[ERRORE] Errore nella classificazione qualita: {e}")
            return {'categoria': 'non_valutabile', 'confidence': 0.0}
    
    def rileva_mobili(self, image_path: str, conf_threshold: float = 0.4) -> List[Dict]:
        """Rileva mobili in un'immagine e classifica il loro stato"""
        try:
            # Carica immagine
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"[ERRORE] Impossibile caricare {image_path}")
                return []
            
            # Detection con YOLO
            results = self.detector(image, conf=conf_threshold, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    # Estrai informazioni
                    classe = result.names[int(box.cls[0])]
                    
                    # Filtra solo classi presenti nel dizionario
                    if classe not in self.dimensioni_standard:
                        continue
                    
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()
                    
                    # Calcola dimensioni bbox
                    bbox_width = bbox[2] - bbox[0]
                    bbox_height = bbox[3] - bbox[1]
                    
                    # Calcola volume
                    volume_info = self.calcola_volume(classe, bbox_width, bbox_height)
                    
                    # Classifica qualita
                    qualita = self.classifica_qualita(image, bbox)
                    
                    detection = {
                        'classe': classe,
                        'sottotipo': volume_info.get('sottotipo', 'unknown'),
                        'confidence': round(confidence, 3),
                        'bbox': {
                            'x1': int(bbox[0]),
                            'y1': int(bbox[1]),
                            'x2': int(bbox[2]),
                            'y2': int(bbox[3])
                        },
                        'qualita': qualita,
                        'volume_stimato': volume_info,
                        'immagine': str(image_path)
                    }
                    detections.append(detection)
                detections = self.filtra_box_sovrapposti(detections, iou_threshold=0.05)
            return detections
            
        except Exception as e:
            print(f"[ERRORE] Errore durante elaborazione {image_path}: {e}")
            return []
    
    def processa_batch(self, images_folder: str, output_folder: str = 'output'):
        """Processa un batch di immagini"""
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        
        # Crea cartella Output_Dataset per i crop
        dataset_path = Path('Output_Dataset')
        dataset_path.mkdir(exist_ok=True)
        
        # Trova immagini
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(images_folder).glob(f"*{ext}"))
        
        print(f"\n[INFO] Trovate {len(image_files)} immagini")
        
        all_results = []
        summary = {
            'totale_oggetti': 0,
            'per_classe': {},
            'per_qualita': {'buono': 0, 'da_aggiustare': 0, 'da_buttare': 0},
            'volume_totale_m3': 0
        }
        
        crop_counter = 0
        
        for idx, img_path in enumerate(image_files, 1):
            print(f"[{idx}/{len(image_files)}] Elaborazione {img_path.name}...")
            
            detections = self.rileva_mobili(str(img_path))
            
            if detections:
                all_results.extend(detections)
                
                # Carica immagine originale
                image = cv2.imread(str(img_path))
                
                # Aggiorna summary e salva crop
                for det in detections:
                    summary['totale_oggetti'] += 1
                    
                    # Per classe
                    classe = det['classe']
                    if classe not in summary['per_classe']:
                        summary['per_classe'][classe] = 0
                    summary['per_classe'][classe] += 1
                    
                    # Per qualita
                    qualita = det['qualita']['categoria']
                    if qualita in summary['per_qualita']:
                        summary['per_qualita'][qualita] += 1
                    
                    # Volume
                    if det['volume_stimato']['volume_m3']:
                        summary['volume_totale_m3'] += det['volume_stimato']['volume_m3']
                    
                    # Salva crop
                    bbox = det['bbox']
                    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                    crop = image[y1:y2, x1:x2]
                    
                    if crop.size > 0:
                        crop_filename = f"{img_path.stem}_{classe}_{qualita}_{crop_counter:04d}.jpg"
                        crop_path = dataset_path / crop_filename
                        cv2.imwrite(str(crop_path), crop)
                        crop_counter += 1
                
                # Salva immagine annotata
                try:
                    annotated_image = image.copy()
                    for det in detections:
                        bbox = det['bbox']
                        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                        
                        # Colore in base alla qualita
                        colors = {
                            'buono': (0, 255, 0),
                            'da_aggiustare': (0, 165, 255),
                            'da_buttare': (0, 0, 255),
                            'non_valutabile': (128, 128, 128)
                        }
                        color = colors.get(det['qualita']['categoria'], (255, 255, 255))
                        
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                        
                        label = f"{det['classe']} - {det['qualita']['categoria']}"
                        cv2.putText(annotated_image, label, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    output_img = output_path / f"annotated_{img_path.name}"
                    cv2.imwrite(str(output_img), annotated_image)
                except Exception as e:
                    print(f"[ERRORE] Impossibile salvare immagine annotata: {e}")
        
        # Salva risultati
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        json_path = output_path / f"results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        summary['volume_totale_m3'] = round(summary['volume_totale_m3'], 3)
        summary_path = output_path / f"summary_{timestamp}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Report console
        print("\n" + "="*50)
        print("RIEPILOGO ELABORAZIONE")
        print("="*50)
        print(f"[STAT] Mobili rilevati: {summary['totale_oggetti']}")
        print(f"[STAT] Volume totale: {summary['volume_totale_m3']} m³")
        print(f"[STAT] Crop salvati: {crop_counter}")
        print("\n[STAT] Per tipologia:")
        for classe, count in summary['per_classe'].items():
            print(f"  • {classe}: {count}")
        print("\n[STAT] Per condizione:")
        for qualita, count in summary['per_qualita'].items():
            print(f"  • {qualita}: {count}")
        print("="*50)
        print(f"\n[OK] Immagini annotate salvate in: {output_path}/")
        print(f"[OK] Crop dataset salvati in: {dataset_path}/")
        
        return all_results


def main():
    """Esempio di utilizzo completo"""
    
    print("\n" + "="*60)
    print("SISTEMA RICONOSCIMENTO MOBILI - YOLOv8 FINE-TUNED")
    print("="*60)
    
    # Inizializza sistema
    system = FurnitureDetectorSystem()
    
    # Menu interattivo
    while True:
        print("\n[MENU] MENU PRINCIPALE:")
        print("1. Rileva mobili da immagini")
        print("2. Fine-tuning detector (richiede dataset)")
        print("3. Training quality classifier (richiede dataset)")
        print("4. Info modelli caricati")
        print("0. Esci")
        
        choice = input("\nScegli opzione: ")
        
        if choice == '1':
            # Rilevamento mobili
            folder = input("Inserisci path cartella immagini [default: immagini_mobili]: ")
            folder = folder if folder else "immagini_mobili"
            
            if not Path(folder).exists():
                print(f"[ERRORE] Cartella {folder} non trovata")
                continue
            
            output = input("Cartella output [default: output]: ")
            output = output if output else "output"
            
            # Processa batch
            all_detections = system.processa_batch(folder, output)
            
            # Opzione GUI
            if all_detections:
                show_gui = input("\nMostrare visualizzatore grafico? (s/n) [default: s]: ")
                if show_gui.lower() != 'n':
                    print("\n[INFO] Apertura visualizzatore...")
                    gui = FurnitureVisualizerGUI(capacita_camion_m3=4.0, detector_system=system)
                    gui.show(all_detections)
            
        elif choice == '2':
            # Fine-tuning detector
            print("\n[!] Per il fine-tuning serve un dataset annotato in formato YOLO")
            print("Struttura richiesta:")
            print("  - images/: cartella con le immagini")
            print("  - labels/: cartella con i file .txt delle annotazioni")
            
            images_dir = input("Path cartella immagini: ")
            labels_dir = input("Path cartella labels: ")
            
            if not Path(images_dir).exists() or not Path(labels_dir).exists():
                print("[ERRORE] Cartelle non trovate")
                continue
            
            # Prepara dataset
            preparer = FurnitureDatasetPreparer()
            yaml_path = preparer.prepare_yolo_dataset(images_dir, labels_dir)
            
            if yaml_path:
                epochs = input("Numero epoche [default: 50]: ")
                epochs = int(epochs) if epochs else 50
                
                system.train_furniture_detector(str(yaml_path), epochs=epochs)
            
        elif choice == '3':
            # Training quality classifier
            print("\n[!] Per il training serve un dataset con labels di qualita")
            print("Formato richiesto:")
            print("  - images/: cartella con immagini ritagliate di mobili")
            print("  - labels.json: file con {\"image.jpg\": \"buono|da_aggiustare|da_buttare\"}")
            
            images_dir = input("Path cartella immagini: ")
            labels_file = input("Path file labels.json: ")
            
            if not Path(images_dir).exists() or not Path(labels_file).exists():
                print("[ERRORE] File/cartelle non trovate")
                continue
            
            epochs = input("Numero epoche [default: 30]: ")
            epochs = int(epochs) if epochs else 30
            
            system.train_quality_classifier(images_dir, labels_file, epochs=epochs)
            
        elif choice == '4':
            # Info modelli
            print("\n[STAT] STATO MODELLI:")
            print(f"• Detector: {'[OK] Fine-tuned' if system.detector_path.exists() else '[!] Base (non fine-tuned)'}")
            print(f"• Quality: {'[OK] Addestrato' if system.quality_path.exists() else '[X] Non addestrato'}")
            print(f"• Device: {system.device}")
            
        elif choice == '0':
            print("\n[BYE] Arrivederci!")
            break
        
        else:
            print("[ERRORE] Opzione non valida")


if __name__ == "__main__":
    main()