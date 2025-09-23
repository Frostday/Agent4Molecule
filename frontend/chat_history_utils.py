import os, json
from datetime import datetime

HISTORY_DIR = "chat_history"
INDEX_FILE = os.path.join(HISTORY_DIR, "conversations.json")
os.makedirs(HISTORY_DIR, exist_ok=True)

def create_conversation(title="New Conversation"):
    conv_id = f"conv_{datetime.utcnow().strftime('%Y-%m-%d_%H-%M')}"
    conv_file = os.path.join(HISTORY_DIR, f"{conv_id}.json")
    with open(conv_file, "w") as f:
        json.dump([], f)
    index = load_index()
    index.append({
        "id": conv_id,
        "title": title,
        "created_at": datetime.utcnow().isoformat(),
        "last_updated": datetime.utcnow().isoformat()
    })
    save_index(index)
    return conv_id

# def load_index():
#     if os.path.exists(INDEX_FILE):
#         with open(INDEX_FILE, "r") as f:
#             return json.load(f)
#     return []

def load_index():
    if os.path.exists(INDEX_FILE):
        try:
            with open(INDEX_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []  # corrupted/empty file → reset
    return []

def save_index(index):
    with open(INDEX_FILE, "w") as f:
        json.dump(index, f, indent=2)

# def save_message(conv_id, role, content):
#     conv_file = os.path.join(HISTORY_DIR, f"{conv_id}.json")
#     with open(conv_file, "r") as f:
#         messages = json.load(f)
#     messages.append({
#         "role": role,
#         "content": content,
#         "timestamp": datetime.utcnow().isoformat()
#     })
#     with open(conv_file, "w") as f:
#         json.dump(messages, f, indent=2)
#     # update index timestamp
#     index = load_index()
#     for conv in index:
#         if conv["id"] == conv_id:
#             conv["last_updated"] = datetime.utcnow().isoformat()
#     save_index(index)

def save_message(conv_id, role, content):
    conv_file = os.path.join(HISTORY_DIR, f"{conv_id}.json")
    if os.path.exists(conv_file):
        with open(conv_file, "r") as f:
            messages = json.load(f)
    else:
        messages = []

    messages.append({
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow().isoformat()
    })

    with open(conv_file, "w") as f:
        json.dump(messages, f, indent=2)

    # update index timestamp
    index = load_index()
    for conv in index:
        if conv["id"] == conv_id:
            conv["last_updated"] = datetime.utcnow().isoformat()
    save_index(index)


def load_conversation(conv_id):
    conv_file = os.path.join(HISTORY_DIR, f"{conv_id}.json")
    if os.path.exists(conv_file):
        with open(conv_file, "r") as f:
            return json.load(f)
    return []