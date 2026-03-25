import argparse
import logging
import json
from pathlib import Path
from typing import Optional

# Import internal modules
from src.logger import setup_logger, get_logger
from src.file_extractor import extract_text_from_file, FileExtractionError

from src.llm_client import LLMClient
from src.history_manager import HistoryManager

from src.models import ConversationHistory

logger = get_logger(__name__)


SYSTEM_PROMPT = """
You are an expert AI assistant specialized in extracting legal company requisites from text documents.
Your task is to analyze the provided text and extract specific fields into a JSON structure.

Output Constraints:
1. Return ONLY valid JSON object. No markdown code blocks (```json), no explanations, no comments.
2. Ensure all required keys are present. If unknown, use null or empty string as appropriate for the field type.
3. Maintain strict schema compliance.

Target Schema:
{
    "full_legal_name": "Full Legal Name",
    "inn": "10-12 digits",
    "ogrn_or_ogrnip": "13 digits",

    "legal_address": "Full Address",
    "signatory": "Name of Signatory",
    "bank_details": {
        "bank_name": "...",
        "bik": "9 digits (optional)",
        "account_number": "20 digits",
        "correspondent_account": "20 digits"
    }
}


If you are in a conversation mode, update only the missing fields based on user input while preserving existing known data.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Agent for Company Requisites Extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )


    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:1234",
        help="LM Studio base URL (default: http://localhost:1234)"
    )

    parser.add_argument(
        "-f", "--file", 
        type=Path, 
        default=None,
        help="Input document path (.pdf, .docx, .xlsx, .txt)"
    )

    parser.add_argument(
        "-o", "--output-history", 
        type=str, 
        default=None,
        help="Custom filename for history JSON (without extension). Auto-generated if omitted."
    )

    return parser.parse_args()


def process_document(client: LLMClient, file_path: Path, history_manager: HistoryManager):
    """Orchestrates document extraction and initial LLM analysis."""
    logger.info(f"Starting processing for file: {file_path}")

    try:
        text_content = extract_text_from_file(file_path)
        
        if not text_content or len(text_content.strip()) < 50:

            logger.warning("File content is too short to be meaningful.")
        
        prompt = f"""Analyze the following document text and extract company requisites. Return ONLY JSON.

Document Content:
{text_content[:2000]}... (content truncated for context)
"""
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        raw_response = client.request_completion(messages, model_name="qwen/qwen3.5-35b-a3b")
        
        # Clean and parse response
        clean_json = LLMClient.clean_json_output(raw_response)
        
        try:
            data_dict = json.loads(clean_json)
            

            if isinstance(data_dict, dict):
                history_manager.update_requisites(data_dict)
                logger.info("Successfully extracted requisites from document.")
            else:
                logger.error(f"LLM returned non-dict JSON structure. Raw: {clean_json[:100]}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from LLM: {e}")
            logger.warning(f"Raw response: {raw_response[:200]}...")


    except FileExtractionError as e:
        logger.critical(f"File Extraction Error: {e}")
    except Exception as e:
        logger.exception(f"Unhandled error during document processing: {e}")


def run_interactive_mode(client: LLMClient, history_manager: HistoryManager):
    """Runs the CLI chat loop."""
    logger.info("Entering interactive chat mode.")

    
    print("\n=== AI Requisite Agent (Interactive Mode) ===")
    print("Type 'exit' or 'quit' to end session.\n")

    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
                
            lower_input = user_input.lower()
            if lower_input in ['exit', 'quit']:
                break
            
            # Prepare context-aware prompt
            current_context_summary = history_manager.get_context_string()
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},

                {"role": "user", "content": f"Current Known Data:\n{current_context_summary}\n\nUser Query: {user_input}"}
            ]

            print("\nProcessing...")
            
            raw_response = client.request_completion(messages, model_name="qwen/qwen3.5-35b-a3b")
            clean_json = LLMClient.clean_json_output(raw_response)
            
            try:
                data_dict = json.loads(clean_json)
                
                if isinstance(data_dict, dict):
                    history_manager.update_requisites(data_dict)
                    print("✓ Data updated in session.")
                    
                    # Print current state summary for user feedback
                    hist = history_manager.load_or_create()
                    if hist.company_details:
                        d = hist.company_details
                        b = d.bank_details
                        print(f"\nCurrent State:\nName: {d.full_legal_name}\nINN: {d.inn}\nBank: {b.bank_name}")
                else:

                    print("⚠ Model returned a non-dict response. Keeping previous data.")
                    
            except json.JSONDecodeError:
                print(f"⚠ Invalid JSON received from model:\n{clean_json[:200]}")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Saving history...")
            break
        except Exception as e:
            logger.error(f"Chat loop error: {e}")



def main():
    # Configure logging at root level before anything else
    setup_logger("ai_agent_root", level=logging.INFO)
    
    args = parse_args()

    try:
        history_manager = HistoryManager(filename=args.output_history)
        

        # Save endpoint info to history context immediately

        hist = history_manager.load_or_create()
        hist.llm_endpoint_used = args.url
        history_manager.save(hist)

        client = LLMClient(args.url)


        if args.file:
            if not args.file.exists():
                logger.error(f"Specified file does not exist: {args.file}")
                return 1
            
            process_document(client, args.file, history_manager)
            
            # After processing a file, offer to switch to chat mode for clarification? 

            # For now, we exit after doc processing as per "extract requisites" goal.
        else:
            run_interactive_mode(client, history_manager)

    except KeyboardInterrupt:
        logger.info("Program interrupted by user.")
    
    return 0



if __name__ == "__main__":
    import sys
    sys.exit(main())

