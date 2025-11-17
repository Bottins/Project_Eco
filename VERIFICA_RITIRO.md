# Funzionalità Verifica Ritiro - Photo Upload Button

## Descrizione
Questa funzionalità permette agli addetti al ritiro di verificare se sono stati aggiunti rifiuti illegalmente vicino all'ingombrante segnalato.

## Caso d'uso
1. L'utente carica una foto dell'ingombrante da ritirare (es. un divano)
2. Il sistema analizza la foto e rileva gli oggetti presenti
3. L'addetto al ritiro, sul posto, fa una nuova foto dello stesso luogo
4. Il sistema confronta le due foto e segnala eventuali oggetti aggiunti

## Come utilizzare

### 1. Avvia il sistema
```bash
python furniture_system.py
```

### 2. Seleziona l'opzione 1 (Rileva mobili)
- Specifica la cartella con le immagini (default: `immagini_mobili`)
- Il sistema processerà le immagini e richiederà se mostrare la GUI

### 3. Nella GUI di visualizzazione
- Naviga tra gli oggetti rilevati usando i pulsanti "Precedente" e "Successivo"
- Per verificare un ritiro, clicca il pulsante **"📸 Verifica Ritiro"**

### 4. Carica la foto di verifica
- Si aprirà una finestra di dialogo per selezionare la foto scattata dall'addetto
- Seleziona la foto e clicca "Apri"

### 5. Analisi automatica
- Il sistema analizzerà la nuova foto con YOLO
- Confronterà gli oggetti rilevati con quelli della foto originale
- Mostrerà i risultati in una finestra dedicata

## Risultati del confronto

La finestra dei risultati mostrerà:

### Immagini affiancate
- **FOTO ORIGINALE**: con tutti gli oggetti rilevati evidenziati in verde
- **FOTO VERIFICA**: con oggetti originali in verde e oggetti aggiunti in rosso (marcati "NUOVO!")

### Riepilogo dettagliato
- Numero totale oggetti nella foto originale vs foto di verifica
- Elenco oggetti per classe in entrambe le foto
- **OGGETTI AGGIUNTI ILLEGALMENTE** (se presenti):
  - Tipo di oggetto
  - Volume stimato
  - Stato qualitativo
  - Volume totale aggiunto
- **OGGETTI RIMOSSI** (ritiro completato):
  - Tipo e quantità di oggetti rimossi

### Alert visivo
- **✓ NESSUN OGGETTO AGGIUNTO** (verde): tutto OK
- **⚠️ ATTENZIONE: N OGGETTO/I AGGIUNTO/I!** (rosso): rifiuti aggiunti illegalmente

## Esempio pratico

### Scenario
1. Foto originale: 1 divano segnalato per il ritiro
2. Foto verifica: 1 divano + 1 frigorifero + 2 sedie

### Risultato
```
⚠️ ATTENZIONE: 3 OGGETTO/I AGGIUNTO/I!

OGGETTI AGGIUNTI ILLEGALMENTE:
  • REFRIGERATOR (medio)
    Volume: 0.600 m³
    Qualità: DA BUTTARE

  • CHAIR (standard)
    Volume: 0.150 m³
    Qualità: BUONO

  • CHAIR (standard)
    Volume: 0.150 m³
    Qualità: DA AGGIUSTARE

VOLUME TOTALE AGGIUNTO: 0.900 m³
```

## Note tecniche

### Come funziona il confronto
1. Il sistema conta il numero di oggetti per classe nella foto originale
2. Rileva gli oggetti nella foto di verifica
3. Confronta i conteggi per classe
4. Gli oggetti in eccesso sono considerati "aggiunti"

### Limitazioni
- Il confronto è basato sul numero di oggetti per classe, non sul matching spaziale
- Se un oggetto viene rimosso e uno diverso aggiunto, potrebbero non essere rilevati
- La precisione dipende dalla qualità delle foto e dall'illuminazione

### Requisiti foto
- Formato: JPG, JPEG, PNG, BMP
- Risoluzione consigliata: almeno 640x480 pixel
- Inquadratura: includere la stessa area della foto originale
- Illuminazione: evitare controluce e ombre eccessive

## Codice rilevante

La funzionalità è implementata in `furniture_system.py`:

- Classe: `FurnitureVisualizerGUI` (linea 163)
- Metodo pulsante: `verifica_ritiro()` (linea 573)
- Metodo confronto: `confronta_oggetti()` (linea 642)
- Metodo visualizzazione: `mostra_risultati_confronto()` (linea 691)

## Supporto

Per problemi o domande, consultare il repository del progetto.
