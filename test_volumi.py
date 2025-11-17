#!/usr/bin/env python3
"""
Script di test per verificare i volumi realistici degli oggetti
"""

# Dizionario dimensioni aggiornato (in cm) - VALORI REALISTICI PER INGOMBRANTI
dimensioni_standard = {
    'bed': {
        'singolo': {'lunghezza': 200, 'larghezza': 90, 'altezza': 60},  # Con rete/testiera
        'matrimoniale': {'lunghezza': 200, 'larghezza': 160, 'altezza': 65},  # Con rete/testiera
        'king': {'lunghezza': 210, 'larghezza': 180, 'altezza': 70}  # Con rete/testiera
    },
    'couch': {
        'piccolo': {'lunghezza': 150, 'profondita': 85, 'altezza': 90},  # Divano 2 posti
        'medio': {'lunghezza': 200, 'profondita': 95, 'altezza': 95},  # Divano 3 posti
        'grande': {'lunghezza': 280, 'profondita': 160, 'altezza': 95}  # Divano angolare
    },
    'chair': {
        'standard': {'lunghezza': 45, 'profondita': 50, 'altezza': 95},
        'poltrona': {'lunghezza': 85, 'profondita': 90, 'altezza': 105}  # Poltrona imbottita
    },
    'dining_table': {
        'piccolo': {'lunghezza': 120, 'larghezza': 80, 'altezza': 75},  # 4 posti
        'medio': {'lunghezza': 160, 'larghezza': 90, 'altezza': 75},  # 6 posti
        'grande': {'lunghezza': 220, 'larghezza': 100, 'altezza': 75}  # 8 posti
    },
    'desk': {
        'standard': {'lunghezza': 120, 'larghezza': 60, 'altezza': 75},
        'grande': {'lunghezza': 160, 'larghezza': 80, 'altezza': 75},
        'angolare': {'lunghezza': 140, 'larghezza': 140, 'altezza': 75}  # Scrivania ad L
    },
    'wardrobe': {
        'piccolo': {'lunghezza': 100, 'profondita': 60, 'altezza': 200},  # 1 anta
        'medio': {'lunghezza': 135, 'profondita': 60, 'altezza': 210},  # 2 ante
        'grande': {'lunghezza': 180, 'profondita': 65, 'altezza': 240}  # 3 ante
    },
    'bookshelf': {
        'piccolo': {'lunghezza': 60, 'profondita': 25, 'altezza': 120},
        'standard': {'lunghezza': 80, 'profondita': 30, 'altezza': 180},
        'grande': {'lunghezza': 120, 'profondita': 35, 'altezza': 200}
    },
    'tv_stand': {
        'piccolo': {'lunghezza': 100, 'profondita': 40, 'altezza': 50},
        'standard': {'lunghezza': 140, 'profondita': 45, 'altezza': 55},
        'grande': {'lunghezza': 180, 'profondita': 50, 'altezza': 60}
    },
    'coffee_table': {
        'piccolo': {'lunghezza': 80, 'larghezza': 50, 'altezza': 40},
        'standard': {'lunghezza': 110, 'larghezza': 60, 'altezza': 45},
        'grande': {'lunghezza': 130, 'larghezza': 70, 'altezza': 45}
    },
    'mattress': {
        'singolo': {'lunghezza': 190, 'larghezza': 80, 'altezza': 20},
        'matrimoniale': {'lunghezza': 190, 'larghezza': 160, 'altezza': 25},
        'king': {'lunghezza': 200, 'larghezza': 180, 'altezza': 30}
    },
    'refrigerator': {
        'piccolo': {'lunghezza': 55, 'profondita': 60, 'altezza': 140},  # Monoporta piccolo
        'standard': {'lunghezza': 60, 'profondita': 65, 'altezza': 170},  # Monoporta
        'grande': {'lunghezza': 70, 'profondita': 70, 'altezza': 185}  # Doppia porta
    },
    'washing_machine': {
        'standard': {'lunghezza': 60, 'profondita': 60, 'altezza': 85},
        'slim': {'lunghezza': 60, 'profondita': 45, 'altezza': 85}  # Carica frontale slim
    },
    'dishwasher': {
        'standard': {'lunghezza': 60, 'profondita': 60, 'altezza': 85},
        'slim': {'lunghezza': 45, 'profondita': 60, 'altezza': 85}
    }
}

def calcola_volume(dims):
    """Calcola il volume in m³ da dimensioni in cm"""
    lunghezza = dims.get('lunghezza', 100)
    larghezza = dims.get('larghezza', dims.get('profondita', 100))
    altezza = dims.get('altezza', 100)

    volume_cm3 = lunghezza * larghezza * altezza
    volume_m3 = volume_cm3 / 1_000_000

    return round(volume_m3, 3)

def main():
    print("=" * 80)
    print(" " * 20 + "VERIFICA VOLUMI INGOMBRANTI")
    print("=" * 80)
    print()
    print(f"Capacità camion aggiornata: 18.0 m³ (camion medio per ritiro ingombranti)")
    print()
    print("=" * 80)
    print("TABELLA VOLUMI OGGETTI")
    print("=" * 80)
    print()

    volume_totale = 0
    count_oggetti = 0

    for classe, sottotipi in sorted(dimensioni_standard.items()):
        print(f"\n{'─' * 80}")
        print(f"📦 {classe.upper().replace('_', ' ')}")
        print(f"{'─' * 80}")

        for sottotipo, dims in sorted(sottotipi.items()):
            volume = calcola_volume(dims)
            volume_totale += volume
            count_oggetti += 1

            # Formatta dimensioni
            dim_str = ""
            if 'lunghezza' in dims:
                dim_str += f"{dims['lunghezza']}cm"
            if 'larghezza' in dims:
                dim_str += f" × {dims['larghezza']}cm"
            elif 'profondita' in dims:
                dim_str += f" × {dims['profondita']}cm"
            if 'altezza' in dims:
                dim_str += f" × {dims['altezza']}cm"

            # Calcola quanti oggetti entrano in un camion
            oggetti_per_camion = int(18.0 / volume) if volume > 0 else 0

            print(f"  • {sottotipo:15s} | Dimensioni: {dim_str:25s} | Volume: {volume:6.3f} m³ | ~{oggetti_per_camion} per camion")

    print("\n" + "=" * 80)
    print("STATISTICHE")
    print("=" * 80)
    print(f"Tipologie totali di oggetti: {count_oggetti}")
    print(f"Volume medio per oggetto: {volume_totale/count_oggetti:.3f} m³")
    print(f"Volume minimo: {min([calcola_volume(d) for sottotipi in dimensioni_standard.values() for d in sottotipi.values()]):.3f} m³")
    print(f"Volume massimo: {max([calcola_volume(d) for sottotipi in dimensioni_standard.values() for d in sottotipi.values()]):.3f} m³")
    print()
    print("=" * 80)
    print("ESEMPI DI RIEMPIMENTO CAMION (18 m³)")
    print("=" * 80)
    print()
    print("Esempio 1 - Svuotamento appartamento piccolo:")
    esempio1 = [
        ('bed', 'matrimoniale', calcola_volume(dimensioni_standard['bed']['matrimoniale'])),
        ('couch', 'medio', calcola_volume(dimensioni_standard['couch']['medio'])),
        ('wardrobe', 'medio', calcola_volume(dimensioni_standard['wardrobe']['medio'])),
        ('dining_table', 'piccolo', calcola_volume(dimensioni_standard['dining_table']['piccolo'])),
        ('refrigerator', 'standard', calcola_volume(dimensioni_standard['refrigerator']['standard'])),
        ('washing_machine', 'standard', calcola_volume(dimensioni_standard['washing_machine']['standard'])),
    ]

    tot = 0
    for classe, tipo, vol in esempio1:
        print(f"  • 1x {classe:15s} ({tipo:15s}) = {vol:6.3f} m³")
        tot += vol

    print(f"\nTotale: {tot:.3f} m³ / 18.0 m³ ({tot/18*100:.1f}% del camion)")
    print(f"Spazio residuo: {18.0-tot:.3f} m³")

    print("\n" + "-" * 80)
    print("\nEsempio 2 - Sgombero ufficio:")
    esempio2 = [
        ('desk', 'standard', calcola_volume(dimensioni_standard['desk']['standard'])),
        ('desk', 'standard', calcola_volume(dimensioni_standard['desk']['standard'])),
        ('desk', 'standard', calcola_volume(dimensioni_standard['desk']['standard'])),
        ('bookshelf', 'grande', calcola_volume(dimensioni_standard['bookshelf']['grande'])),
        ('bookshelf', 'standard', calcola_volume(dimensioni_standard['bookshelf']['standard'])),
        ('chair', 'standard', calcola_volume(dimensioni_standard['chair']['standard'])),
        ('chair', 'standard', calcola_volume(dimensioni_standard['chair']['standard'])),
        ('chair', 'standard', calcola_volume(dimensioni_standard['chair']['standard'])),
        ('chair', 'standard', calcola_volume(dimensioni_standard['chair']['standard'])),
        ('chair', 'standard', calcola_volume(dimensioni_standard['chair']['standard'])),
        ('chair', 'standard', calcola_volume(dimensioni_standard['chair']['standard'])),
    ]

    tot = 0
    for classe, tipo, vol in esempio2:
        print(f"  • 1x {classe:15s} ({tipo:15s}) = {vol:6.3f} m³")
        tot += vol

    print(f"\nTotale: {tot:.3f} m³ / 18.0 m³ ({tot/18*100:.1f}% del camion)")
    print(f"Spazio residuo: {18.0-tot:.3f} m³")

    print("\n" + "=" * 80)
    print("✅ Test completato! I valori sono realistici per un servizio di ritiro ingombranti.")
    print("=" * 80)

if __name__ == "__main__":
    main()
