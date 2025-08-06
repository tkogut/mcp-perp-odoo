#!/usr/bin/env python3
"""
Comprehensive test suite for Intelligent Agent
"""

import asyncio
import os
import sys
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

async def test_environment():
    """Test environment setup"""
    print("🔧 Testing environment setup...")
    
    required_vars = ["PERPLEXITY_API_KEY", "ODOO_PASSWORD"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print(f"❌ Missing environment variables: {missing}")
        return False
    
    print("✅ Environment variables OK")
    return True

async def test_agent_initialization():
    """Test agent can be created"""
    print("🤖 Testing agent initialization...")
    
    try:
        from intelligent_agent import IntelligentPerplexityOdooAgent
        agent = IntelligentPerplexityOdooAgent()
        print("✅ Agent initialized successfully")
        return True, agent
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return False, None

async def test_question_analysis():
    """Test question analysis functionality"""
    print("🔍 Testing question analysis...")
    
    try:
        from intelligent_agent import IntelligentPerplexityOdooAgent
        agent = IntelligentPerplexityOdooAgent()
        
        test_questions = [
            "Jaki był obrót w Q2 2025?",
            "Ile mamy pracowników?",
            "Pokaż produkty magazynowe"
        ]
        
        for question in test_questions:
            analysis = await agent.analyze_question(question)
            print(f"  📝 '{question}' -> modules: {analysis.get('modules', [])}")
        
        print("✅ Question analysis working")
        return True
    except Exception as e:
        print(f"❌ Question analysis failed: {e}")
        return False

async def test_odoo_connection():
    """Test Odoo MCP connection"""
    print("🔗 Testing Odoo MCP connection...")
    
    try:
        from intelligent_agent import IntelligentPerplexityOdooAgent
        agent = IntelligentPerplexityOdooAgent()
        
        # Test basic company data query
        analysis = {"modules": ["company"], "time_range": {}}
        result = await agent.query_by_module("company", analysis)
        
        if result and 'res.company' in result:
            print("✅ Odoo connection working")
            return True
        else:
            print("❌ Odoo connection failed - no data returned")
            return False
            
    except Exception as e:
        print(f"❌ Odoo connection failed: {e}")
        return False

async def test_full_workflow():
    """Test complete agent workflow"""
    print("🔄 Testing full workflow...")
    
    try:
        from intelligent_agent import IntelligentPerplexityOdooAgent
        agent = IntelligentPerplexityOdooAgent()
        
        # Simple question that should work
        question = "Jakie firmy są w systemie?"
        answer = await agent.ask(question)
        
        if answer and len(answer) > 100:
            print("✅ Full workflow working")
            print(f"📄 Answer length: {len(answer)} characters")
            return True
        else:
            print("❌ Full workflow failed - short/empty answer")
            return False
            
    except Exception as e:
        print(f"❌ Full workflow failed: {e}")
        return False

async def run_all_tests():
    """Run comprehensive test suite"""
    print("🧪 INTELLIGENT AGENT TEST SUITE")
    print("=" * 50)
    
    tests = [
        ("Environment", test_environment),
        ("Agent Init", test_agent_initialization),  
        ("Question Analysis", test_question_analysis),
        ("Odoo Connection", test_odoo_connection),
        ("Full Workflow", test_full_workflow)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔬 {test_name.upper()}")
        print("-" * 30)
        
        try:
            result = await test_func()
            success = result if isinstance(result, bool) else result[0]
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Agent is ready for use.")
    else:
        print("⚠️  Some tests failed. Check configuration and dependencies.")
    
    return results

if __name__ == "__main__":
    asyncio.run(run_all_tests())
