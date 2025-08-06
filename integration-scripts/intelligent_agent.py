#!/usr/bin/env python3
"""
FIXED Dynamic Intelligent Odoo Agent
Corrected company filtering and query generation
"""

import asyncio
import json
import re
import os
import sys
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import MCP components
sys.path.append(str(Path(__file__).parent))
from integrate_mcp import MCPClient


class FixedDynamicOdooAgent:
    """Fixed dynamic agent with proper company filtering"""
    
    def __init__(self):
        self.validate_environment()
        
        # Available Perplexity models
        self.available_models = [
            "sonar-pro", "sonar", "sonar-reasoning", "sonar-medium-online"
        ]
        
        # Odoo connection settings
        self.odoo_config = {
            "ODOO_URL": "https://test.miw.group",
            "ODOO_DB": "test", 
            "ODOO_USERNAME": "tomasz.kogut@miw.group",
            "ODOO_PASSWORD": os.getenv("ODOO_PASSWORD")
        }
    
    def validate_environment(self):
        """Validate required API keys"""
        required = {
            "PERPLEXITY_API_KEY": "Get from: https://www.perplexity.ai/settings/api",
            "ODOO_PASSWORD": "Your Odoo password"
        }
        
        missing = [f"❌ {k}: {v}" for k, v in required.items() if not os.getenv(k)]
        if missing:
            raise ValueError("Missing environment variables:\n" + "\n".join(missing))
        
        print("✅ Environment validation passed")
    
    async def query_perplexity(self, prompt: str, model: str = "sonar-pro") -> Dict[str, Any]:
        """Query Perplexity API for intelligent analysis"""
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert Odoo ERP analyst. Analyze user questions and generate precise Odoo API queries. Return structured JSON responses only."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 2000
        }
        
        headers = {
            "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}",
            "Content-Type": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.perplexity.ai/chat/completions",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content']
                        return {"success": True, "data": content}
                    else:
                        error_text = await response.text()
                        return {"success": False, "error": f"HTTP {response.status}: {error_text}"}
        
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def find_all_companies(self) -> Dict[str, int]:
        """FIXED: Get all companies in system first"""
        
        client = MCPClient(
            server_path=str(Path(__file__).parent / "mcp-odoo" / "run_server.py"),
            env=self.odoo_config
        )
        
        companies = {}
        
        try:
            await client.start()
            await client.initialize()
            
            result = await client.call_tool("execute_method", {
                "model": "res.company",
                "method": "search_read",
                "kwargs": {
                    "domain": [],
                    "fields": ['id', 'name'],
                    "limit": 50
                }
            })
            
            if isinstance(result, dict) and 'content' in result:
                content_text = result['content'][0].get('text', '')
                data = json.loads(content_text)
                
                if data.get('success') and data.get('result'):
                    for company in data['result']:
                        companies[company['name']] = company['id']
                        
            print(f"🏢 Found companies in system:")
            for name, company_id in companies.items():
                print(f"   - ID: {company_id}, Name: {name}")
                        
        finally:
            await client.close()
        
        return companies
    
    async def find_best_matching_company(self, search_name: str) -> Optional[Tuple[str, int]]:
        """FIXED: Find best matching company with fuzzy matching"""
        
        companies = await self.find_all_companies()
        
        if not companies:
            return None
        
        search_lower = search_name.lower()
        
        # Exact match first
        for name, company_id in companies.items():
            if name.lower() == search_lower:
                print(f"🎯 Exact match found: {name} (ID: {company_id})")
                return (name, company_id)
        
        # Partial match
        for name, company_id in companies.items():
            if any(word in name.lower() for word in search_lower.split()):
                print(f"🎯 Partial match found: {name} (ID: {company_id})")
                return (name, company_id)
        
        # Fuzzy match for common patterns
        for name, company_id in companies.items():
            if "miw" in search_lower and "miw" in name.lower():
                print(f"🎯 Fuzzy match found: {name} (ID: {company_id})")
                return (name, company_id)
        
        print(f"⚠️ No company found matching: {search_name}")
        return None
    
    async def analyze_question_with_fixed_ai(self, question: str) -> Dict[str, Any]:
        """FIXED: Use AI to analyze question with better company handling"""
        
        analysis_prompt = f"""
        Analyze this business question about Odoo ERP and provide query strategy.
        
        QUESTION: "{question}"
        
        Extract and return ONLY valid JSON:
        {{
            "companies": ["company names mentioned or empty array"],
            "time_period": {{
                "start_date": "YYYY-MM-DD or null",
                "end_date": "YYYY-MM-DD or null",
                "description": "human readable period"
            }},
            "models_to_query": [
                {{
                    "model": "account.move or sale.order",
                    "purpose": "why this model",
                    "base_filters": [["field", "operator", "value"]],
                    "fields": ["field1", "field2"],
                    "aggregation_needed": true/false
                }}
            ],
            "query_intent": "what user wants to know",
            "financial_focus": "revenue|profit|orders|invoices"
        }}
        
        Rules:
        - For "obrót" (turnover/revenue) use account.move with out_invoice and posted state
        - Extract company names exactly as written
        - Q1 = Jan-Mar, Q2 = Apr-Jun, Q3 = Jul-Sep, Q4 = Oct-Dec
        """
        
        try:
            result = await self.query_perplexity(analysis_prompt, model="sonar-pro")
            
            if result.get('success'):
                json_match = re.search(r'\{.*\}', result['data'], re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                    return self._validate_analysis(analysis, question)
            
            return self._fallback_analysis(question)
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return self._fallback_analysis(question)
    
    def _validate_analysis(self, analysis: Dict[str, Any], question: str) -> Dict[str, Any]:
        """Validate and enhance AI analysis"""
        
        # Ensure required fields
        if not analysis.get('models_to_query'):
            analysis['models_to_query'] = [{
                "model": "account.move",
                "fields": ["name", "amount_total", "invoice_date"],
                "base_filters": []
            }]
        
        # Fix date nulls
        time_period = analysis.get('time_period', {})
        if time_period.get('start_date') == "null":
            time_period['start_date'] = None
        if time_period.get('end_date') == "null":
            time_period['end_date'] = None
        
        return analysis
    
    def _fallback_analysis(self, question: str) -> Dict[str, Any]:
        """Fallback analysis if AI fails"""
        
        q = question.lower()
        
        # Extract companies
        companies = []
        company_patterns = [
            r'(MIW\s+Group[^,]*)',
            r'(Labs\d+[^,]*)',
            r'([A-Z][a-zA-Z\s]*\s+sp\.\s*z\.?\s*o\.\s*o\.)',
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            companies.extend(matches)
        
        # Extract time period
        time_period = {"start_date": None, "end_date": None, "description": "all time"}
        if 'q1 2025' in q:
            time_period = {"start_date": "2025-01-01", "end_date": "2025-03-31", "description": "Q1 2025"}
        elif 'q2 2025' in q:
            time_period = {"start_date": "2025-04-01", "end_date": "2025-06-30", "description": "Q2 2025"}
        
        return {
            "companies": companies,
            "time_period": time_period,
            "models_to_query": [{
                "model": "account.move",
                "fields": ["name", "amount_total", "invoice_date", "partner_id"],
                "base_filters": []
            }],
            "query_intent": "revenue analysis",
            "financial_focus": "revenue"
        }
    
    async def execute_fixed_queries(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """FIXED: Execute queries with proper company filtering"""
        
        client = MCPClient(
            server_path=str(Path(__file__).parent / "mcp-odoo" / "run_server.py"),
            env=self.odoo_config
        )
        
        results = {}
        
        try:
            await client.start()
            await client.initialize()
            
            # Find company ID if specific company mentioned
            company_id = None
            company_name = None
            
            companies = analysis.get('companies', [])
            if companies:
                search_name = companies[0]
                match_result = await self.find_best_matching_company(search_name)
                if match_result:
                    company_name, company_id = match_result
                else:
                    print(f"⚠️ Company '{search_name}' not found, using all companies")
            
            # Process each model query
            for model_query in analysis['models_to_query']:
                model = model_query['model']
                
                print(f"🔍 Querying {model}")
                
                # Build clean domain
                domain = []
                
                # Add time filters
                time_period = analysis.get('time_period', {})
                if time_period.get('start_date') and time_period.get('end_date'):
                    if model == 'account.move':
                        domain.extend([
                            ['invoice_date', '>=', time_period['start_date']],
                            ['invoice_date', '<=', time_period['end_date']],
                            ['state', '=', 'posted'],
                            ['move_type', '=', 'out_invoice']
                        ])
                    elif model == 'sale.order':
                        domain.extend([
                            ['date_order', '>=', time_period['start_date']],
                            ['date_order', '<=', time_period['end_date']],
                            ['state', 'in', ['sale', 'done']]
                        ])
                
                # Add company filter ONLY if company found
                if company_id and model in ['account.move', 'sale.order']:
                    domain.append(['company_id', '=', company_id])
                    print(f"🏢 Using company filter: {company_name} (ID: {company_id})")
                else:
                    print(f"🌐 No company filter - querying all companies")
                
                print(f"📋 Domain: {domain}")
                
                # Get count
                try:
                    count_result = await client.call_tool("execute_method", {
                        "model": model,
                        "method": "search_count",
                        "args": [domain]
                    })
                    
                    count = 0
                    if isinstance(count_result, dict) and 'content' in count_result:
                        content_text = count_result['content'][0].get('text', '')
                        parsed_count = json.loads(content_text)
                        if parsed_count.get('success'):
                            count = parsed_count.get('result', 0)
                    
                    print(f"📊 Found {count} records in {model}")
                
                except Exception as e:
                    print(f"📊 Count failed for {model}: {e}")
                    count = 0
                
                # Get data
                limit = min(count, 1000) if count > 0 else 100
                
                result = await client.call_tool("execute_method", {
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
        
        return totals
    
    async def ask(self, question: str) -> str:
        """FIXED main method"""
        
        print(f"🤖 Processing question with FIXED AI: {question}")
        
        try:
            # 1. FIXED AI analysis
            print("🧠 Analyzing question with Perplexity AI...")
            analysis = await self.analyze_question_with_fixed_ai(question)
            
            companies = ', '.join(analysis.get('companies', [])) or "All companies"
            time_desc = analysis.get('time_period', {}).get('description', 'all time')
            models = [mq['model'] for mq in analysis.get('models_to_query', [])]
            
            print(f"🎯 FIXED AI Analysis Results:")
            print(f"   Companies: {companies}")
            print(f"   Time period: {time_desc}")
            print(f"   Models to query: {models}")
            print(f"   Intent: {analysis.get('query_intent', 'Unknown')}")
            
            # 2. FIXED query execution
            print("📊 Executing FIXED Odoo queries...")
            results = await self.execute_fixed_queries(analysis)
            
            # 3. Calculate totals
            print("🔢 Calculating financial totals...")
            totals = self.calculate_fixed_totals(results)
            
            # 4. Generate response
            print("✨ Generating response...")
            
            response = f"""## Odpowiedź na pytanie: {question}

**Wyniki dla: {companies}**
**Okres: {time_desc}**

### Podsumowanie finansowe
- **Łączny obrót**: {totals['grand_total']:,.2f} PLN
- **Faktury**: {totals['invoice_total']:,.2f} PLN
- **Zamówienia**: {totals['order_total']:,.2f} PLN
- **Liczba rekordów**: {totals['record_count']:,}

### Szczegółowe dane
{chr(10).join(totals['details']) or 'Brak danych do wyświetlenia'}

### Metodologia FIXED
- **Wyszukiwanie firm**: Najpierw pobiera wszystkie firmy z systemu, następnie używa fuzzy matching
- **Filtrowanie**: {'Zastosowano filtr company_id' if any('company_filter' in r and r['company_filter'] != 'All companies' for r in results.values()) else 'Brak filtra - wszystkie firmy'}
- **Domeny**: Czyste, bez duplikatów
- **Modele AI**: sonar-pro dla analizy

### Status
{'✅ ZNALEZIONO DANE' if totals['grand_total'] > 0 else '⚠️ BRAK DANYCH - sprawdź czy firma istnieje i ma faktury w tym okresie'}
"""
            
            return response
            
        except Exception as e:
            logger.error(f"FIXED agent failed: {e}")
            return f"## Błąd w FIXED Dynamic Agent\n\n❌ **Błąd**: {str(e)}"


async def main():
    """Main function"""
    
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        try:
            agent = FixedDynamicOdooAgent()
            answer = await agent.ask(question)
            print(f"\n📋 FIXED AGENT RESPONSE:\n{answer}")
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        try:
            agent = FixedDynamicOdooAgent()
            
            print("🚀 FIXED Dynamic Intelligent Odoo Agent Ready!")
            print("✨ FIXED Features:")
            print("  🏢 Proper company discovery and matching")
            print("  🧹 Clean domain generation (no duplicates)")
            print("  🎯 Fuzzy company name matching")
            print("  📊 All companies fallback")
            print("\nType 'quit' to exit.\n")
            
            while True:
                try:
                    question = input("❓ Your business question: ").strip()
                    if question.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    if question:
                        answer = await agent.ask(question)
                        print(f"\n📋 RESPONSE:\n{answer}\n")
                        print("-" * 80)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
            
            print("👋 Goodbye!")
            
        except Exception as e:
            print(f"❌ Failed to initialize agent: {e}")


if __name__ == "__main__":
    asyncio.run(main())
