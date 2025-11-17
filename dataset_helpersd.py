"""
Helper scripts per preparare dataset per il training
- Creazione dataset di esempio
- Auto-labeling con YOLO base
- Utilities per annotazione manuale
"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import shutil
from typing import List, Dict

# Setup encoding per caratteri speciali
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass


class DatasetCreator:
    """Crea un dataset di esempio o aiuta nella preparazione"""
    
    def __init__(self, base_dir: str = "dataset_preparation"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    def auto_label_with_yolo(self, images_dir: str, output_dir: str = "auto_labeled"):
        """
        Auto-etichetta immagini usando YOLO base per creare dataset iniziale
        """
        output_path = self.base_dir / output_dir
        (output_path / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Carica YOLO base
        print("[INFO] Caricamento YOLOv8 per auto-labeling...")
        model = YOLO('yolov8m.pt')
        
        # Classi mobili da rilevare
        furniture_classes = [
            'bed', 'couch', 'chair', 'dining table', 'tv', 
            'laptop', 'refrigerator', 'oven', 'sink', 'toilet'
        ]
        
        # Mappa le classi COCO ai nostri nomi
        coco_to_furniture = {
            'bed': 'bed',
            'couch': 'couch', 
            'chair': 'chair',
            'dining table': 'dining_table',
            'tv': 'tv_stand',
            'refrigerator': 'refrigerator',
            'oven': 'oven',
            'sink': 'sink',
            'toilet': 'toilet'
        }
        
        image_files = list(Path(images_dir).glob("*.jpg")) + list(Path(images_dir).glob("*.png"))
        print(f"[INFO] Trovate {len(image_files)} immagini")
        
        valid_images = 0
        
        for img_path in image_files:
            try:
                # Rileva oggetti
                results = model(str(img_path), verbose=False)
                
                # Prepara annotazioni YOLO format
                labels = []
                has_furniture = False
                
                for result in results:
                    boxes = result.boxes
                    if boxes is None:
                        continue
                        
                    for box in boxes:
                        class_name = result.names[int(box.cls[0])]
                        
                        if class_name in coco_to_furniture:
                            has_furniture = True
                            # Converti bbox in formato YOLO (normalized xywh)
                            bbox = box.xywh[0].cpu().numpy()
                            img_h, img_w = result.orig_shape
                            
                            x_center = bbox[0] / img_w
                            y_center = bbox[1] / img_h
                            width = bbox[2] / img_w
                            height = bbox[3] / img_h
                            
                            # Usa l'indice della classe mappata
                            furniture_name = coco_to_furniture[class_name]
                            class_idx = list(coco_to_furniture.values()).index(furniture_name)
                            
                            labels.append(f"{class_idx} {x_center} {y_center} {width} {height}")
                
                # Salva solo se contiene mobili
                if has_furniture:
                    # Copia immagine
                    shutil.copy2(img_path, output_path / 'images' / img_path.name)
                    
                    # Salva labels
                    label_path = output_path / 'labels' / f"{img_path.stem}.txt"
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(labels))
                    
                    valid_images += 1
                    print(f"  [OK] {img_path.name} - {len(labels)} mobili trovati")
                else:
                    print(f"  [SKIP] {img_path.name} - nessun mobile")
                    
            except Exception as e:
                print(f"  [ERRORE] {img_path.name}: {e}")
        
        print(f"\n[OK] Auto-labeling completato!")
        print(f"   {valid_images}/{len(image_files)} immagini con mobili")
        print(f"   Salvate in: {output_path}")
        
        return str(output_path)
    
    def create_quality_dataset(self, cropped_images_dir: str):
        """
        Crea dataset per quality classifier partendo da immagini ritagliate
        """
        output_path = self.base_dir / 'quality_dataset'
        output_path.mkdir(exist_ok=True)
        
        # Crea sottocartelle per categoria
        categories = ['buono', 'da_aggiustare', 'da_buttare']
        for cat in categories:
            (output_path / cat).mkdir(exist_ok=True)
        
        print("\n" + "="*50)
        print("CREAZIONE DATASET QUALITA'")
        print("="*50)
        print("Sposta manualmente le immagini nelle cartelle:")
        for cat in categories:
            print(f"  • {output_path / cat}/ per mobili '{cat}'")
        
        print("\nOppure usa il Quality Labeler interattivo (opzione 2)")
        
        return str(output_path)


class QualityLabeler:
    """GUI per etichettare rapidamente la qualità dei mobili"""
    
    def __init__(self, images_dir: str, output_file: str = "quality_labels.json"):
        self.images_dir = Path(images_dir)
        self.output_file = output_file
        self.current_index = 0
        self.finished = False  # Flag per indicare se abbiamo finito
        
        # Carica immagini
        self.image_files = list(self.images_dir.glob("*.jpg")) + \
                          list(self.images_dir.glob("*.png"))
        
        # Carica o inizializza labels
        if Path(output_file).exists():
            with open(output_file, 'r') as f:
                self.labels = json.load(f)
        else:
            self.labels = {}
        
        if not self.image_files:
            print("[ERRORE] Nessuna immagine trovata!")
            return
        
        # Crea GUI
        self.create_gui()
    
    def create_gui(self):
        """Crea interfaccia grafica per labeling"""
        self.root = tk.Tk()
        self.root.title("Quality Labeler - Classificazione Mobili")
        self.root.geometry("800x700")
        
        # Configura chiusura finestra
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Frame principale
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Info frame
        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=0, column=0, columnspan=3, pady=5)
        
        self.info_label = ttk.Label(info_frame, text="", font=('Arial', 12))
        self.info_label.pack()
        
        # Canvas per immagine
        self.canvas = tk.Canvas(main_frame, width=600, height=400, bg='gray')
        self.canvas.grid(row=1, column=0, columnspan=3, pady=10)
        
        # Frame bottoni qualità
        quality_frame = ttk.Frame(main_frame)
        quality_frame.grid(row=2, column=0, columnspan=3, pady=10)
        
        ttk.Label(quality_frame, text="Seleziona qualità:", font=('Arial', 11)).pack()
        
        buttons_frame = ttk.Frame(quality_frame)
        buttons_frame.pack(pady=5)
        
        # Bottoni qualità con colori
        self.btn_buono = tk.Button(
            buttons_frame, text="BUONO (1)", bg='green', fg='white',
            font=('Arial', 12, 'bold'), width=15, height=2,
            command=lambda: self.set_quality('buono')
        )
        self.btn_buono.grid(row=0, column=0, padx=5)
        
        self.btn_aggiustare = tk.Button(
            buttons_frame, text="DA AGGIUSTARE (2)", bg='orange', fg='white',
            font=('Arial', 12, 'bold'), width=15, height=2,
            command=lambda: self.set_quality('da_aggiustare')
        )
        self.btn_aggiustare.grid(row=0, column=1, padx=5)
        
        self.btn_buttare = tk.Button(
            buttons_frame, text="DA BUTTARE (3)", bg='red', fg='white',
            font=('Arial', 12, 'bold'), width=15, height=2,
            command=lambda: self.set_quality('da_buttare')
        )
        self.btn_buttare.grid(row=0, column=2, padx=5)
        
        # Frame navigazione
        nav_frame = ttk.Frame(main_frame)
        nav_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        self.btn_prev = ttk.Button(nav_frame, text="← Precedente", command=self.prev_image)
        self.btn_prev.pack(side=tk.LEFT, padx=5)
        
        self.btn_next = ttk.Button(nav_frame, text="Successiva →", command=self.next_image)
        self.btn_next.pack(side=tk.LEFT, padx=5)
        
        self.btn_skip = ttk.Button(nav_frame, text="Salta (Spazio)", command=self.skip_image)
        self.btn_skip.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, length=600, mode='determinate')
        self.progress.grid(row=4, column=0, columnspan=3, pady=10)
        
        # Stato corrente
        self.status_label = ttk.Label(main_frame, text="", font=('Arial', 10))
        self.status_label.grid(row=5, column=0, columnspan=3)
        
        # Statistiche
        self.stats_label = ttk.Label(main_frame, text="", font=('Arial', 9))
        self.stats_label.grid(row=6, column=0, columnspan=3, pady=5)
        
        # Salva ed esci
        self.btn_save_exit = ttk.Button(main_frame, text="💾 Salva ed Esci", 
                  command=self.save_and_exit)
        self.btn_save_exit.grid(row=7, column=1, pady=20)
        
        # Shortcuts tastiera
        self.root.bind('1', lambda e: self.set_quality('buono') if not self.finished else None)
        self.root.bind('2', lambda e: self.set_quality('da_aggiustare') if not self.finished else None)
        self.root.bind('3', lambda e: self.set_quality('da_buttare') if not self.finished else None)
        self.root.bind('<Left>', lambda e: self.prev_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.bind('<space>', lambda e: self.skip_image())
        self.root.bind('<Control-s>', lambda e: self.save_labels())
        
        # Carica prima immagine
        self.load_image()
        self.update_stats()
        
        self.root.mainloop()
    
    def load_image(self):
        """Carica immagine corrente"""
        if self.current_index >= len(self.image_files):
            self.show_completion()
            return
        
        img_path = self.image_files[self.current_index]
        
        try:
            # Carica e ridimensiona immagine
            image = Image.open(img_path)
            image.thumbnail((600, 400), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(image)
            
            # Mostra su canvas
            self.canvas.delete("all")
            self.canvas.create_image(300, 200, image=self.photo)
            
            # Aggiorna info
            self.info_label.config(
                text=f"Immagine {self.current_index + 1}/{len(self.image_files)}: {img_path.name}"
            )
            
            # Aggiorna progress bar
            self.progress['value'] = (self.current_index / len(self.image_files)) * 100
            
            # Mostra stato corrente
            current_label = self.labels.get(img_path.name, "Non etichettata")
            self.status_label.config(text=f"Stato attuale: {current_label}")
            
            # Evidenzia bottone se già etichettata
            self.reset_button_colors()
            if current_label == 'buono':
                self.btn_buono.config(relief=tk.SUNKEN, bg='dark green')
            elif current_label == 'da_aggiustare':
                self.btn_aggiustare.config(relief=tk.SUNKEN, bg='dark orange')
            elif current_label == 'da_buttare':
                self.btn_buttare.config(relief=tk.SUNKEN, bg='dark red')
                
            # Abilita/disabilita bottone precedente
            self.btn_prev.config(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
            
        except Exception as e:
            print(f"[ERRORE] Impossibile caricare {img_path}: {e}")
            self.skip_image()
    
    def show_completion(self):
        """Mostra messaggio di completamento e disabilita controlli"""
        self.finished = True
        
        # Disabilita tutti i bottoni di qualità
        self.btn_buono.config(state=tk.DISABLED)
        self.btn_aggiustare.config(state=tk.DISABLED)
        self.btn_buttare.config(state=tk.DISABLED)
        self.btn_next.config(state=tk.DISABLED)
        self.btn_skip.config(state=tk.DISABLED)
        
        # Mostra messaggio
        self.info_label.config(text="✅ COMPLETATO! Tutte le immagini sono state processate!")
        self.status_label.config(text="Premi 'Salva ed Esci' per terminare")
        
        # Pulisci canvas
        self.canvas.delete("all")
        self.canvas.create_text(
            300, 200, 
            text="Fine immagini!\n\nTutte le immagini sono state processate.\nPremi 'Salva ed Esci' per salvare i risultati.",
            font=('Arial', 14),
            fill='green',
            justify='center'
        )
        
        # Progress bar al 100%
        self.progress['value'] = 100
        
        # Auto-salvataggio dopo 3 secondi
        self.root.after(3000, self.auto_save_prompt)
    
    def auto_save_prompt(self):
        """Chiede se salvare automaticamente"""
        if self.finished:
            result = messagebox.askyesno(
                "Salvataggio automatico",
                "Vuoi salvare ed uscire?\n\n" + 
                f"Etichettate: {len(self.labels)}/{len(self.image_files)} immagini"
            )
            if result:
                self.save_and_exit()
    
    def reset_button_colors(self):
        """Reset colori bottoni"""
        self.btn_buono.config(relief=tk.RAISED, bg='green')
        self.btn_aggiustare.config(relief=tk.RAISED, bg='orange')
        self.btn_buttare.config(relief=tk.RAISED, bg='red')
    
    def set_quality(self, quality: str):
        """Imposta qualità per immagine corrente"""
        if self.finished or self.current_index >= len(self.image_files):
            return
            
        img_name = self.image_files[self.current_index].name
        self.labels[img_name] = quality
        print(f"[OK] {img_name} -> {quality}")
        
        # Aggiorna statistiche
        self.update_stats()
        
        # Controlla se è l'ultima immagine
        if self.current_index == len(self.image_files) - 1:
            # Vai alla schermata di completamento
            self.current_index += 1
            self.load_image()
        else:
            # Altrimenti passa alla prossima
            self.next_image()
    
    def next_image(self):
        """Vai all'immagine successiva"""
        if self.current_index < len(self.image_files):
            self.current_index += 1
            self.load_image()
    
    def prev_image(self):
        """Vai all'immagine precedente"""
        if self.current_index > 0:
            self.current_index -= 1
            self.finished = False  # Reset flag se torniamo indietro
            
            # Riabilita bottoni se erano disabilitati
            self.btn_buono.config(state=tk.NORMAL)
            self.btn_aggiustare.config(state=tk.NORMAL)
            self.btn_buttare.config(state=tk.NORMAL)
            self.btn_next.config(state=tk.NORMAL)
            self.btn_skip.config(state=tk.NORMAL)
            
            self.load_image()
    
    def skip_image(self):
        """Salta immagine senza etichettare"""
        if not self.finished:
            self.next_image()
    
    def update_stats(self):
        """Aggiorna statistiche etichettatura"""
        total = len(self.image_files)
        labeled = len(self.labels)
        
        # Conta per categoria
        counts = {'buono': 0, 'da_aggiustare': 0, 'da_buttare': 0}
        for label in self.labels.values():
            if label in counts:
                counts[label] += 1
        
        stats_text = (
            f"Totale: {labeled}/{total} | "
            f"Buono: {counts['buono']} | "
            f"Da aggiustare: {counts['da_aggiustare']} | "
            f"Da buttare: {counts['da_buttare']}"
        )
        
        self.stats_label.config(text=stats_text)
    
    def save_labels(self):
        """Salva labels su file"""
        try:
            with open(self.output_file, 'w') as f:
                json.dump(self.labels, f, indent=2)
            print(f"[SAVE] Labels salvate in {self.output_file}")
            return True
        except Exception as e:
            print(f"[ERRORE] Impossibile salvare: {e}")
            messagebox.showerror("Errore", f"Impossibile salvare il file:\n{e}")
            return False
    
    def on_closing(self):
        """Gestisce chiusura finestra"""
        if len(self.labels) > 0:
            result = messagebox.askyesnocancel(
                "Conferma chiusura",
                f"Hai {len(self.labels)} etichette.\nVuoi salvare prima di uscire?"
            )
            if result is None:  # Cancel
                return
            elif result:  # Yes - salva
                if not self.save_labels():
                    return
        self.root.destroy()
    
    def save_and_exit(self):
        """Salva labels ed esci"""
        if self.save_labels():
            labeled = len(self.labels)
            total = len(self.image_files)
            
            print(f"\n[OK] Salvate {labeled}/{total} etichette in {self.output_file}")
            
            # Mostra riepilogo finale
            messagebox.showinfo(
                "Completato",
                f"Etichettatura completata!\n\n"
                f"Etichettate: {labeled}/{total} immagini\n"
                f"Salvate in: {self.output_file}"
            )
            
            self.root.destroy()


class DatasetSplitter:
    """Divide dataset esistenti in train/val/test"""
    
    @staticmethod
    def split_yolo_dataset(dataset_dir: str, train_ratio: float = 0.7, 
                           val_ratio: float = 0.2, test_ratio: float = 0.1):
        """
        Divide un dataset YOLO in train/val/test
        """
        dataset_path = Path(dataset_dir)
        images_path = dataset_path / 'images'
        labels_path = dataset_path / 'labels'
        
        if not images_path.exists() or not labels_path.exists():
            print("[ERRORE] Dataset non valido. Servono cartelle 'images' e 'labels'")
            return
        
        # Crea struttura output
        output_path = dataset_path / 'split_dataset'
        for split in ['train', 'val', 'test']:
            (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Raccogli file
        image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
        np.random.shuffle(image_files)
        
        # Calcola split
        n = len(image_files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        splits = {
            'train': image_files[:n_train],
            'val': image_files[n_train:n_train + n_val],
            'test': image_files[n_train + n_val:]
        }
        
        # Copia file
        for split_name, split_files in splits.items():
            for img_path in split_files:
                try:
                    # Immagine
                    shutil.copy2(img_path, output_path / split_name / 'images' / img_path.name)
                    
                    # Label
                    label_path = labels_path / f"{img_path.stem}.txt"
                    if label_path.exists():
                        shutil.copy2(label_path, output_path / split_name / 'labels' / label_path.name)
                except Exception as e:
                    print(f"[ERRORE] Impossibile copiare {img_path.name}: {e}")
            
            print(f"[OK] {split_name}: {len(split_files)} immagini")
        
        print(f"\n[INFO] Dataset diviso salvato in: {output_path}")


def main():
    """Menu principale per tools dataset"""
    
    print("\n" + "="*60)
    print("DATASET PREPARATION TOOLS")
    print("="*60)
    
    while True:
        print("\n[MENU] STRUMENTI DISPONIBILI:")
        print("1. Auto-labeling con YOLO (crea dataset iniziale)")
        print("2. Quality Labeler GUI (classifica stato mobili)")
        print("3. Split dataset (train/val/test)")
        print("4. Crea struttura dataset vuota")
        print("0. Esci")
        
        choice = input("\nScegli opzione: ")
        
        if choice == '1':
            # Auto-labeling
            creator = DatasetCreator()
            images_dir = input("Path cartella immagini da etichettare: ")
            
            if not Path(images_dir).exists():
                print("[ERRORE] Cartella non trovata")
                continue
            
            output_dir = input("Nome cartella output [default: auto_labeled]: ")
            output_dir = output_dir if output_dir else "auto_labeled"
            
            creator.auto_label_with_yolo(images_dir, output_dir)
            
        elif choice == '2':
            # Quality labeler GUI
            images_dir = input("Path cartella immagini ritagliate di mobili: ")
            
            if not Path(images_dir).exists():
                print("[ERRORE] Cartella non trovata")
                continue
            
            output_file = input("Nome file output [default: quality_labels.json]: ")
            output_file = output_file if output_file else "quality_labels.json"
            
            print("\n[INFO] Avvio GUI per etichettatura...")
            print("Shortcuts:")
            print("  1/2/3: Etichetta come buono/da aggiustare/da buttare")
            print("  ←/→: Naviga tra immagini")
            print("  Spazio: Salta immagine")
            print("  Ctrl+S: Salva labels\n")
            
            labeler = QualityLabeler(images_dir, output_file)
            
        elif choice == '3':
            # Split dataset
            dataset_dir = input("Path dataset da dividere (con images/ e labels/): ")
            
            if not Path(dataset_dir).exists():
                print("[ERRORE] Cartella non trovata")
                continue
            
            print("\nRatio split (devono sommare a 1.0):")
            try:
                train = float(input("  Train ratio [default: 0.7]: ") or "0.7")
                val = float(input("  Val ratio [default: 0.2]: ") or "0.2")
                test = float(input("  Test ratio [default: 0.1]: ") or "0.1")
                
                if abs(train + val + test - 1.0) > 0.01:
                    print("[ERRORE] I ratio devono sommare a 1.0")
                    continue
                    
                DatasetSplitter.split_yolo_dataset(dataset_dir, train, val, test)
            except ValueError:
                print("[ERRORE] Inserisci numeri validi")
            
        elif choice == '4':
            # Crea struttura vuota
            base_dir = input("Nome cartella base [default: nuovo_dataset]: ")
            base_dir = base_dir if base_dir else "nuovo_dataset"
            
            base_path = Path(base_dir)
            (base_path / 'images').mkdir(parents=True, exist_ok=True)
            (base_path / 'labels').mkdir(parents=True, exist_ok=True)
            
            print(f"[OK] Creata struttura in {base_path}/")
            print("   Inserisci immagini in 'images/' e annotazioni in 'labels/'")
            
        elif choice == '0':
            print("\n[BYE] Arrivederci!")
            break
        
        else:
            print("[ERRORE] Opzione non valida")


if __name__ == "__main__":
    main()