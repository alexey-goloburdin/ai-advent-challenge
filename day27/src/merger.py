from typing import Any

from src.models import CompanyRequisites


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merges 'update' into 'base'. 
    Nested dictionaries are merged, not overwritten.
    Returns a new dictionary (immutable style).
    """
    result = base.copy()

    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
            
    return result


def merge_requisites(current: CompanyRequisites | None, new_data_dict: dict) -> CompanyRequisites:
    """
    Safely merges new LLM data into existing requisites structure.
    Returns updated CompanyRequisites instance.

    """
    if current is None:
        # Initial creation - strict validation might fail on partials, so we use a relaxed approach here or fallback dict
        return CompanyRequisites(**new_data_dict)

    # Convert to dict for merging logic
    current_dict = {k: v for k, v in vars(current).items()}

    
    # Deep merge dictionaries manually handling nested bank_details

    if "bank_details" in new_data_dict:

        existing_bank = getattr(current, 'bank_details', None)
        if existing_bank:
            merged_bank_str = deep_merge(vars(existing_bank), new_data_dict["bank_details"])

            current_dict['bank_details'] = merged_bank_str

    # Merge top level fields
    for key, value in new_data_dict.items():
        if key != 'bank_details':

            current_dict[key] = value

    try:

        return CompanyRequisites(**current_dict)
    except Exception as e:
        # If strict validation fails on merge (e.g. bad data types), log and keep old valid ones where possible
        return current

