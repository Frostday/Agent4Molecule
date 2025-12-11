import os, json
from datetime import datetime
from typing import Any,Dict

# STATE_DIR = "agent_states"

class ChatState: 


    def __init__(self,parent_dir): #if conv_id exists, load from file #if conv_id doesn't exist, start new one 
        # self.conv_id = conv_id
        # os.makedirs(STATE_DIR, exist_ok = True)
        self.parent_dir = parent_dir
        # self.state_path = os.path.join(STATE_DIR, f"{self.conv_id}.json")
        
        # if os.path.exists(self.state_path):
        #     self._load_state()
        
        # else:
        #     self._create_new_state()
    
    def _getState(self,conv_id):
        state_path = os.path.join(self.parent_dir + "/" + conv_id, f"{conv_id}_agent_state.json")

        if os.path.exists(state_path):
            return self._load_state(state_path)
        
        else:
            return self._create_new_state(conv_id)


    def writeState(self,conv_id,state): #write current state to file 
        state["last_updated"] = datetime.now().isoformat()
        state_path = os.path.join(self.parent_dir + "/" + conv_id, f"{conv_id}_agent_state.json")
        with open(state_path,"w") as fh:
            json.dump(state, fh, indent=2)


    def getState(self, conv_id):
        state = self._getState(conv_id)
        return state
    
    def update_task(self,conv_id,task_id,key,value): #write new information to state
        state = self._getState(conv_id)
        state['task_history'][str(task_id)][key] = value
        
        self.writeState(conv_id,state)
    
    def get(self,conv_id,key,default=None):
        state = self._getState(conv_id)
        return state.get(key,default)
    


    def create_new_task(self,conv_id,user_query): #will have to add files at some point
        state = self._getState(conv_id)
        task_id = len(state['task_history']) + 1 #some random task id
        task = {
            "task_id": str(task_id),
            "query": user_query,
            "created_at": datetime.now().isoformat(),
            "status": "in progress",
            "execution_history": [] #some kind of list
        }

        state["task_history"][str(task_id)] = task
        self.writeState(conv_id,state)
        return task_id
 
    def add_agent_response(self,conv_id,task_id,agent_response_obj):
        state = self._getState(conv_id)
        # print(agent_response_obj)
        # d = {"type": "agent response", "response": agent_response_obj.candidates[0].content.parts[0].text}
        d = {
    "type": "agent response",
    "response": agent_response_obj["content"]
        }
    def add_tool_call(self,conv_id,task_id,tool_name,tool_args,tool_id):
        state = self._getState(conv_id)
        d = {"type": "tool call", "tool_name": tool_name, "tool_args": tool_args,"tool_id": tool_id }
        state['task_history'][str(task_id)]['execution_history'].append(d)
        self.writeState(conv_id,state)    

    def add_tool_response(self,conv_id,task_id,tool_name,tool_obj):
        state = self._getState(conv_id)
        d = {"type": "tool response", "tool_name": tool_name, "tool_execution_status": tool_obj['status'], "tool_result" : tool_obj['answer']}
        state['task_history'][str(task_id)]['execution_history'].append(d)
        self.writeState(conv_id,state)



    def _load_state(self,state_path):
        state= {}
        # state_path = os.path.join(self.parent_dir + "/" + conv_id, f"{conv_id}_agent_state.json")
        # print("state_path",state_path)
        with open(state_path,"r") as fh:
            state = json.load(fh)
        

        # print("state",state)
        return state

    # def _load_state(self, state_path):
    #     with open(state_path, "r") as fh:
    #         state = fh.read()   # returns raw string contents
    #     return state

    def _create_new_state(self,conv_id):
        state  = {
            "conv_id" : conv_id,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "flags": {"awaiting_user": False, "tool_busy": False},
            "task_history" : {}
          
        }

        return state
    
    def __getitem__(self, conv_id,key):
        state = self._getState(conv_id)
        return state[key]
    
    