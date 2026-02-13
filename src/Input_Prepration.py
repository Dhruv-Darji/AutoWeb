"""
SeeAct Input Preparation Module

Take values:
- single task action containing instruction, screenshot, cleaned HTML etc.
- history of previous actions for the task (if needed for input preparation)
And prepare the input for the Action Generation step. This include:
- website screenshot
- instruction (user given high level instruction)
- fixed prompt template (for now, can be more dynamic in the future)
- past history of actions

The output of this module will be the input for the Action Generation step, which will be a vision-language (Qwen-2.4-VL-2B-Instruct) model for SeeAct.
"""

from typing import Dict, List, Tuple, Optional

class SeeActInputPreparator:    
    def __init__(self):
        pass

    def prepare_input(self,
                      action: Dict,
                      history: List[Dict]) -> Dict:
        """
        Prepare input for a single action generation step.
        
        Args:
            action: Dict containing details of the current action (including instruction, website screenshot, cleaned HTML, etc.)
            history: List of previous Action Generation planes/steps in the task (if needed for input preparation)
        Returns:
            Dict containing prepared inputs for action generation, including:
            - "screenshot": website screenshot to be given in Action Generation
            - "prompt": the prompt text to be given in Action Generation (including instruction, history, etc.)
        """
        
        instruction = action.get("instruction", "")

        website_screenshot = action.get("screenshot", None)

        prompt_text = f"""
You are an expert web automation assistant. 
Your task is to generate textual action plans to accomplish the user's task based on the instruction and the website screenshot provided.

Instruction (What user want to achieve): {instruction},

What previous actions planned and executed so far: {history}

Based on the above instruction, website screenshot and history, please generate the next action plan to move towards accomplishing the user's task.
"""
        return {
            "screenshot": website_screenshot,
            "prompt": prompt_text
        }