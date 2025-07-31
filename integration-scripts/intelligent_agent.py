#!/usr/bin/env python3
"""
Intelligent Perplexity-Odoo Agent with Dynamic Model Selection
"""

import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any

class IntelligentPerplexityOdooAgent:
    
    def __init__(self, perplexity_integrator, odoo_client):
        self.perplexity = perplexity_integrator
        self.odoo_client = odoo_client
        
        # Mapa modułów biznesowych na modele Odoo
        self.model_mapping = {
            'finance': ['account.move', 'account.move.line'],
            'sales': ['sale.order', 'sale.order.line'],
            'crm': ['crm.lead', 'res.partner'],
            'hr': ['hr.employee', 'hr.attendance', 'hr.holidays'],
            'inventory': ['product.product', 'stock.quant', 'stock.move'],
            'company': ['res.company', 'res.users'],
            'project': ['project.project', 'project.task']
        }
    
    async def analyze_question(self, question: str) -> Dict[str, Any]:
        """Analizuje pytanie i określa strategię zapytań"""
        
        analysis_prompt = f"""
        Przeanalizuj następujące pytanie biznesowe i określ:
        
        1. Jakie moduły Odoo są potrzebne: finance, sales, crm, hr, inventory, company, project
        2. Jaki zakres czasowy (jeśli dotyczy): konkretne daty, okresy
        3. Jakie konkretne dane są potrzebne
        
        Pytanie: "{question}"
        
        Odpowiedz w formacie JSON:
        {{
            "modules": ["module1", "module2"],
            "time_range": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}},
            "data_focus": "opis potrzebnych danych",
            "complexity": "simple|medium|complex"
        }}
        """
        
        response = await self.perplexity.query_perplexity_direct(analysis_prompt)
        
        try:
            # Parsuj odpowiedź JSON z Perplexity
            analysis = json.loads(self.extract_json_from_text(response['data']))
            return analysis
        except:
            # Fallback - prosta analiza na podstawie słów kluczowych
            return self.fallback_analysis(question)
    
    def extract_json_from_text(self, text: str) -> str:
        """Wyciąga JSON z tekstu odpowiedzi"""
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text)
        return matches[0] if matches else '{}'
    
    def fallback_analysis(self, question: str) -> Dict[str, Any]:
        """Prosta analiza oparta na słowach kluczowych"""
        question_lower = question.lower()
        
        modules = []
        if any(word in question_lower for word in ['obrót', 'przychód', 'faktura', 'sprzedaż', 'finanse']):
            modules.extend(['finance', 'sales'])
        if any(word in question_lower for word in ['klient', 'lead', 'możliwość', 'crm']):
            modules.append('crm')
        if any(word in question_lower for word in ['pracownik', 'urlop', 'hr', 'zespół']):
            modules.append('hr')
        if any(word in question_lower for word in ['magazyn', 'produkt', 'stock']):
            modules.append('inventory')
        
        return {
            "modules": modules or ['company'],  # Domyślnie dane firmy
            "time_range": self.extract_time_range(question),
            "data_focus": "general business data",
            "complexity": "medium"
        }
    
    def extract_time_range(self, question: str) -> Dict[str, str]:
        """Wyciąga zakres czasowy z pytania"""
        question_lower = question.lower()
        
        if 'q3 2025' in question_lower:
            return {"start": "2025-07-01", "end": "2025-09-30"}
        elif '2025' in question_lower:
            return {"start": "2025-01-01", "end": "2025-12-31"}
        elif 'ostatni miesiąc' in question_lower:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            return {"start": start_date.strftime('%Y-%m-%d'), "end": end_date.strftime('%Y-%m-%d')}
        
        return {"start": None, "end": None}
    
    async def query_by_module(self, module: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Wykonuje zapytania dla konkretnego modułu"""
        
        models = self.model_mapping.get(module, [])
        module_results = {}
        
        for model in models:
            try:
                # Buduj dynamiczną domenę na podstawie analizy
                domain = self.build_domain_for_model(model, analysis)
                fields = self.get_relevant_fields(model, analysis)
                
                result = await self.odoo_client.call_tool("execute_method", {
                    "model": model,
                    "method": "search_read",
                    "kwargs": {
                        "domain": domain,
                        "fields": fields,
                        "limit": 20
                    }
                })
                
                module_results[model] = result
                
            except Exception as e:
                module_results[model] = {"error": str(e)}
        
        return module_results
    
    def build_domain_for_model(self, model: str, analysis: Dict[str, Any]) -> List:
        """Buduje dynamiczną domenę filtracji dla modelu"""
        domain = []
        
        time_range = analysis.get('time_range', {})
        start_date = time_range.get('start')
        end_date = time_range.get('end')
        
        # Filtry czasowe dla różnych modeli
        if start_date and end_date:
            if model == 'account.move':
                domain.extend([
                    ["invoice_date", ">=", start_date],
                    ["invoice_date", "<=", end_date],
                    ["state", "=", "posted"]
                ])
            elif model == 'sale.order':
                domain.extend([
                    ["date_order", ">=", start_date],
                    ["date_order", "<=", end_date],
                    ["state", "in", ["sale", "done"]]
                ])
            elif model == 'crm.lead':
                domain.extend([
                    ["create_date", ">=", start_date],
                    ["create_date", "<=", end_date]
                ])
        
        # Dodatkowe filtry specyficzne dla modeli
        if model == 'account.move':
            domain.append(["move_type", "=", "out_invoice"])  # Tylko faktury sprzedażowe
        elif model == 'hr.employee':
            domain.append(["active", "=", True])  # Tylko aktywni pracownicy
        
        return domain
    
    def get_relevant_fields(self, model: str, analysis: Dict[str, Any]) -> List[str]:
        """Określa jakie pola pobrać dla modelu"""
        
        field_mapping = {
            'account.move': ['name', 'partner_id', 'amount_total', 'invoice_date', 'state'],
            'sale.order': ['name', 'partner_id', 'amount_total', 'date_order', 'state'],
            'crm.lead': ['name', 'partner_id', 'probability', 'expected_revenue', 'stage_id'],
            'res.partner': ['name', 'email', 'phone', 'country_id', 'is_company'],
            'hr.employee': ['name', 'department_id', 'job_title', 'work_email'],
            'product.product': ['name', 'list_price', 'qty_available', 'categ_id'],
            'res.company': ['name', 'email', 'website', 'phone'],
            'res.users': ['name', 'login', 'email', 'active']
        }
        
        return field_mapping.get(model, ['name', 'id'])
    
    async def synthesize_intelligent_answer(self, question: str, analysis: Dict[str, Any], all_data: Dict[str, Any]) -> str:
        """Syntetyzuje inteligentną odpowiedź na podstawie zebranych danych"""
        
        # Przygotuj strukturalne podsumowanie danych
        data_summary = self.prepare_data_summary(all_data)
        
        synthesis_prompt = f"""
        Na podstawie zebranych danych z systemu Odoo MIW Group, odpowiedz na pytanie biznesowe.
        
        **Pytanie:** {question}
        
        **Analiza pytania:** {analysis}
        
        **Zebrane dane z Odoo:**
        {data_summary}
        
        **Instrukcje:**
        1. Odpowiedz konkretnie na zadane pytanie
        2. Użyj rzeczywistych danych z systemu
        3. Podaj konkretne liczby, kwoty, daty jeśli są dostępne
        4. Wskaż źródła danych (z jakich modeli Odoo)
        5. Dodaj strategiczne wnioski biznesowe
        6. Jeśli brakuje danych, jasno to wskaż
        
        Sformatuj odpowiedź w strukturze:
        ## Odpowiedź na pytanie
        ## Kluczowe dane
        ## Źródła danych Odoo
        ## Wnioski strategiczne
        """
        
        response = await self.perplexity.query_perplexity_direct(synthesis_prompt)
        return response['data']
    
    def prepare_data_summary(self, all_data: Dict[str, Any]) -> str:
        """Przygotowuje strukturalne podsumowanie danych"""
        summary = []
        
        for module, module_data in all_data.items():
            summary.append(f"\n**Moduł {module}:**")
            
            for model, model_data in module_data.items():
                if isinstance(model_data, dict) and 'content' in model_data:
                    try:
                        content_text = model_data['content'][0].get('text', '')
                        parsed_data = json.loads(content_text)
                        
                        if parsed_data.get('success') and 'result' in parsed_data:
                            results = parsed_data['result']
                            summary.append(f"  - {model}: {len(results)} rekordów")
                            
                            # Dodaj próbkę danych
                            if results:
                                first_record = results[0]
                                sample_fields = {k: v for k, v in first_record.items() if k in ['name', 'amount_total', 'date_order', 'invoice_date']}
                                summary.append(f"    Przykład: {sample_fields}")
                        
                    except:
                        summary.append(f"  - {model}: błąd parsowania danych")
                else:
                    summary.append(f"  - {model}: brak danych lub błąd")
        
        return "\n".join(summary)
    
    async def ask(self, question: str) -> str:
        """Główna metoda agenta - odpowiada na pytanie biznesowe"""
        
        print(f"🤖 Analizuję pytanie: {question}")
        
        # 1. Analizuj pytanie
        analysis = await self.analyze_question(question)
        print(f"🔍 Identyfikowane moduły: {analysis.get('modules', [])}")
        
        # 2. Zbierz dane z odpowiednich modułów
        all_data = {}
        for module in analysis.get('modules', []):
            print(f"📊 Pobieram dane z modułu: {module}")
            all_data[module] = await self.query_by_module(module, analysis)
        
        # 3. Syntetyzuj odpowiedź
        print(f"🧠 Generuję inteligentną odpowiedź...")
        answer = await self.synthesize_intelligent_answer(question, analysis, all_data)
        
        return answer

# Przykład integracji z istniejącym systemem
async def create_intelligent_agent():
    """Tworzy instancję inteligentnego agenta"""
    
    # Użyj istniejących komponentów
    from integrate_mcp import MCPIntegrator
    
    # Inicjalizuj istniejący integrator
    integrator = MCPIntegrator()
    
    # Stwórz klienta Odoo MCP (używa istniejący kod)
    from integrate_mcp import MCPClient
    import os
    from pathlib import Path
    
    script_dir = Path(__file__).parent
    odoo_server_path = script_dir / "mcp-odoo" / "run_server.py"
    
    odoo_client = MCPClient(
        server_path=str(odoo_server_path),
        env={
            "ODOO_URL": "https://test.miw.group",
            "ODOO_DB": "test",
            "ODOO_USERNAME": "tomasz.kogut@miw.group", 
            "ODOO_PASSWORD": os.getenv("ODOO_PASSWORD", "")
        }
    )
    
    # Stwórz inteligentnego agenta
    agent = IntelligentPerplexityOdooAgent(integrator, odoo_client)
    
    return agent

# Przykład użycia
async def main():
    agent = await create_intelligent_agent()
    
    # Przykładowe pytania biznesowe
    questions = [
        "Jaki był obrót MIW Group w Q2 2025?"
    ]
    
    for question in questions:
        print(f"\n{'='*60}")
        answer = await agent.ask(question)
        print(f"Q: {question}")
        print(f"A: {answer}")

if __name__ == "__main__":
    asyncio.run(main())
