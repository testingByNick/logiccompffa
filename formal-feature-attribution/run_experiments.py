#!/usr/bin/env python3
"""
Script principal para executar todos os experimentos
"""

import argparse
from src.experiments.section_5_1 import run_section_5_1
from src.experiments.section_5_2 import run_section_5_2
from src.utils.visualization import generate_final_report

def main():
    parser = argparse.ArgumentParser(description='Reprodução dos experimentos')
    parser.add_argument('--section', choices=['5.1', '5.2', 'all'], default='all')
    parser.add_argument('--output-dir', default='data/results/')
    
    args = parser.parse_args()
    
    if args.section in ['5.1', 'all']:
        print("Executando experimentos da Seção 5.1")
        results_5_1 = run_section_5_1()
    
    if args.section in ['5.2', 'all']:
        print("Executando experimentos da Seção 5.2")
        results_5_2 = run_section_5_2(results_5_1)
    
    generate_final_report(results_5_1, results_5_2, args.output_dir)

if __name__ == "__main__":
    main()