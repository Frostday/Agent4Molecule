import os, json
from datetime import datetime


class ChatHistory():

    def __init__(self,user_dir):
        self.user_dir = user_dir
        self.index_file = os.path.join(self.user_dir, "conversations.json")

    def init_history(self,conv_id,title="New Conversation"):
        
        index = self.load_index()
       
        title += " " +  datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(title)
        index.append({
            "id": conv_id,
            "title": title,
            "created_at": datetime.utcnow().isoformat(),
            "last_updated": datetime.utcnow().isoformat()
        })

        self.save_index(index)
        

    def load_index(self):
        print("inside load index")
    
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return []  # corrupted/empty file → reset
        return []

    def save_index(self,index):
        with open(self.index_file, "w") as f:
            json.dump(index, f, indent=2)


    def save_message(self,conv_id, role, content):
        conv_dir = self.user_dir + "/" + conv_id
        conv_file = os.path.join(conv_dir, f"{conv_id}.json")
        if os.path.exists(conv_file):
            with open(conv_file, "r") as f:
                messages = json.load(f)
        else:
            messages = []

        print("inside user_dir: ", role, content)
        messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })
        print("messages",messages)
        print(conv_file)

        with open(conv_file, "w") as f:
            json.dump(messages, f, indent=2)

    # update index timestamp
        index = self.load_index()
        for conv in index:
            if conv["id"] == conv_id:
                conv["last_updated"] = datetime.utcnow().isoformat()
        # print("save_message",index)
        self.save_index(index)


    def load_conversation(self,conv_id):
        conv_dir = self.user_dir + "/" + conv_id
        conv_file = os.path.join(conv_dir, f"{conv_id}.json")
        if os.path.exists(conv_file):
            with open(conv_file, "r") as f:
                return json.load(f)
        return []

    def delete_conversation(self,conv_id):
        conv_dir = self.user_dir + "/" + conv_id
        """Delete a conversation JSON file and remove it from the index."""
        conv_file = os.path.join(conv_dir, f"{conv_id}.json")
        if os.path.exists(conv_file):
            os.remove(conv_file)

    # Update index
        index = self.load_index()
        index = [conv for conv in index if conv["id"] != conv_id]
        print("del",index)
        self.save_index(index)
