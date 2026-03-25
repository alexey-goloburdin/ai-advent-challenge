import json
from pathlib import Path
from datetime import datetime
import uuid
from typing import Optional

from src.models import ConversationHistory, ChatMessage, CompanyRequisites
from src.merger import merge_requisites
from src.logger import get_logger


logger = get_logger(__name__)


class HistoryManager:
    def __init__(self, history_dir: str = "history", filename: Optional[str] = None):
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)

        if filename:
            self.file_path = self.history_dir / f"{filename}.json"
        else:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            self.file_path = self.history_dir / f"history_{timestamp_str}_{unique_id}.json"

    def load_or_create(self) -> ConversationHistory:
        if self.file_path.exists():
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:

                    data = json.load(f)
                    return ConversationHistory.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load history file {self.file_path}, creating new session. Error: {e}")
        
        # Create fresh history if not found or corrupted
        return ConversationHistory()

    def save(self, history: ConversationHistory):
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(history.to_dict(), f, ensure_ascii=False, indent=2)
            

            logger.info(f"Session saved successfully to {self.file_path}")
        except Exception as e:

            logger.error(f"Failed to save history file: {e}")


    def update_requisites(self, new_data_dict: dict):
        """Updates company details in the current session state."""
        history = self.load_or_create()
        

        try:
            if history.company_details is None:
                # First extraction or empty state - create object directly from dict
                # Note: In a real robust system, we might validate against CompanyRequisites here strictly
                # but for partial data flow, we pass through.
                history.company_details = CompanyRequisites(**new_data_dict)
            else:
                # Merge new data into existing structure preserving known fields
                history.company_details = merge_requisites(history.company_details, new_data_dict)


            self.save(history)
            logger.info("Company requisites updated in session state.")
        except Exception as e:
            logger.error(f"Error updating company details: {e}")

            logger.warning("Keeping previous valid data.")

    def get_context_string(self) -> str:
        """Generates a summary string to send back to the LLM for context."""

        if self.load_or_create().company_details is None:
            return "No requisites found yet. The user needs to provide information."
        
        details = self.load_or_create().company_details

        
        # Format current state concisely
        bank = details.bank_details
        summary = (
            f"--- CURRENT KNOWN DATA ---\n"
            f"Legal Name: {details.full_legal_name}\n"

            f"INN: {details.inn}\n"
            f"OGRN: {details.ogrn_or_ogrnip}\n"
            f"Address: {details.legal_address}\n"
            f"Signatory: {details.signatory}\n"
            f"Bank Name: {bank.bank_name}\n"
            f"Account: {bank.account_number}\n"
        )
        return summary

    def add_message(self, role: str, content: str):
        history = self.load_or_create()
        history.messages.append(ChatMessage(role=role, content=content))

