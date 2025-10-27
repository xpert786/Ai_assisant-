#!/usr/bin/env python
"""
Test script to create a new company record in Airtable Accounts table
"""

import os
import sys
import django

# Setup Django environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'n8n.settings')
django.setup()

from app.utils import AirtableClient
from django.utils import timezone

def test_create_account():
    """Test function to create a new account in Airtable"""
    
    print("\n" + "="*80)
    print("TEST: Create Account in Airtable")
    print("="*80 + "\n")
    
    # Initialize Airtable client
    try:
        print("[INIT] Initializing Airtable client...")
        airtable_client = AirtableClient()
        print("[INIT] Airtable client initialized successfully\n")
    except Exception as e:
        print(f"[INIT] ERROR: Failed to initialize Airtable client: {str(e)}")
        return False
    
    # Test data - Using only Company_Name (the only required field that exists in Accounts table)
    test_company_name = "Test Company " + timezone.now().strftime("%Y%m%d%H%M%S")
    test_data = {
        'Company_Name': test_company_name
    }
    
    print(f"[TEST] Test company name: {test_company_name}")
    print(f"[TEST] Test data: {test_data}\n")
    
    # Step 1: Check if account already exists
    print("-" * 80)
    print("STEP 1: Check if account already exists")
    print("-" * 80)
    try:
        existing_account = airtable_client.search_accounts_by_name(test_company_name)
        if existing_account:
            print(f"[STEP 1] FAILED: Account '{test_company_name}' already exists with ID: {existing_account.get('id')}")
            print(f"[STEP 1] Exiting test to avoid creating duplicate")
            return False
        else:
            print(f"[STEP 1] SUCCESS: Account '{test_company_name}' does not exist - proceeding with creation")
    except Exception as e:
        print(f"[STEP 1] ERROR: Exception while checking for existing account: {str(e)}")
        return False
    
    # Step 2: Create the account
    print("\n" + "-" * 80)
    print("STEP 2: Create new account record")
    print("-" * 80)
    
    try:
        print(f"[STEP 2] Attempting to create account with data: {test_data}")
        result = airtable_client.create_record('Accounts', test_data)
        
        if result and isinstance(result, dict):
            if result.get('error'):
                print(f"[STEP 2] FAILED: Account creation returned error: {result}")
                if result.get('status_code') == 422:
                    print(f"[STEP 2] VALIDATION ERROR: {result.get('details', 'No details provided')}")
                return False
            elif result.get('id'):
                print(f"[STEP 2] SUCCESS: Account created successfully!")
                print(f"[STEP 2] Account ID: {result.get('id')}")
                print(f"[STEP 2] Full response: {result}")
                return True
            else:
                print(f"[STEP 2] UNKNOWN: Unexpected response format: {result}")
                return False
        else:
            print(f"[STEP 2] FAILED: Invalid response: {result}")
            return False
            
    except Exception as e:
        print(f"[STEP 2] ERROR: Exception during account creation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_create_client():
    """Test function to create a new client in Airtable"""
    
    print("\n" + "="*80)
    print("TEST: Create Client in Airtable")
    print("="*80 + "\n")
    
    # Initialize Airtable client
    try:
        print("[INIT] Initializing Airtable client...")
        airtable_client = AirtableClient()
        print("[INIT] Airtable client initialized successfully\n")
    except Exception as e:
        print(f"[INIT] ERROR: Failed to initialize Airtable client: {str(e)}")
        return False
    
    # Test data
    test_client_name = "Test Client " + timezone.now().strftime("%Y%m%d%H%M%S")
    test_contact = "test.contact@example.com"
    test_message = "Test message from script"
    
    print(f"[TEST] Test client name: {test_client_name}")
    print(f"[TEST] Test contact: {test_contact}\n")
    
    # Check if client already exists
    print("-" * 80)
    print("STEP 1: Check if client already exists")
    print("-" * 80)
    try:
        existing_client = airtable_client.search_clients_by_name(test_client_name)
        if existing_client:
            print(f"[STEP 1] FAILED: Client '{test_client_name}' already exists with ID: {existing_client.get('id')}")
            return False
        else:
            print(f"[STEP 1] SUCCESS: Client '{test_client_name}' does not exist - proceeding")
    except Exception as e:
        print(f"[STEP 1] ERROR: Exception while checking for existing client: {str(e)}")
        return False
    
    # Create the client
    print("\n" + "-" * 80)
    print("STEP 2: Create new client record")
    print("-" * 80)
    
    try:
        print(f"[STEP 2] Creating client record...")
        result = airtable_client.create_client_record(test_client_name, test_contact, test_message)
        
        if result and isinstance(result, dict) and not result.get('error') and result.get('id'):
            print(f"[STEP 2] SUCCESS: Client created successfully!")
            print(f"[STEP 2] Client ID: {result.get('id')}")
            return True
        else:
            print(f"[STEP 2] FAILED: {result}")
            return False
            
    except Exception as e:
        print(f"[STEP 2] ERROR: Exception during client creation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_search_accounts():
    """Test function to search accounts"""
    
    print("\n" + "="*80)
    print("TEST: Search Accounts in Airtable")
    print("="*80 + "\n")
    
    try:
        airtable_client = AirtableClient()
        
        # Get all accounts
        print("[TEST] Fetching all accounts from Airtable...")
        accounts = airtable_client.get_table_records('Accounts')
        print(f"[TEST] Found {len(accounts)} total accounts")
        
        if accounts:
            print("\n[TEST] Sample accounts (first 5):")
            for i, account in enumerate(accounts[:5], 1):
                fields = account.get('fields', {})
                company_name = fields.get('Company_Name', 'N/A')
                account_id = account.get('id', 'N/A')
                print(f"  {i}. {company_name} (ID: {account_id})")
        
        return True
        
    except Exception as e:
        print(f"[TEST] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# AIRTABLE TEST SCRIPT")
    print("# Testing Account and Client Creation")
    print("#"*80)
    
    # Run tests
    results = []
    
    # Test 1: Search accounts
    results.append(("Search Accounts", test_search_accounts()))
    
    # Test 2: Create account
    results.append(("Create Account", test_create_account()))
    
    # Test 3: Create client
    results.append(("Create Client", test_create_client()))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    print("="*80 + "\n")
