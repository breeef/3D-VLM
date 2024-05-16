import random

def format_object_locations(object_locations):
    """
    Formats a dictionary of object names and their corresponding coordinates
    into a specific string format.

    Parameters:
    object_locations (dict): A dictionary where keys are object names and values are tuples of coordinates.

    Returns:
    str: A formatted string representing objects and their coordinates.
    """
    # 创建一个空列表来存储格式化后的对象位置字符串
    formatted_locations = []
    
    # 遍历字典中的每个键值对
    for object_name, coordinates in object_locations.items():
        # 格式化字符串并添加到列表中
        formatted_locations.append(f"{object_name}: {coordinates}")
    
    # 将所有格式化后的位置合并为单一字符串，每个位置后有逗号和空格分隔
    return ", ".join(formatted_locations)
class InstructionTemplateForGPT:
    def __init__(self):
        
        
        self.endprompt = ' You can include a simpletask decomposition, but the length of the decomposition must not exceed 3.'
        # Verification: 验证任务完成情况
        self.verification = {
    "instruction": "Place the cup in the middle drawer.",
    "num frame": "20",
    "obj_loc": {
      "cup": "(0.1, 0.2, 0.3)",
      "drawer": "(0.4, 0.5, 0.6)"
    },
    "task": "Verification",
    "example": [
      {
        "Q": "The initial scene is <initial scene> and the current scene is <current scene>. Has the robot placed the cup in the middle drawer by the 10th frame?",
        "A": "No"
      }
    ]
  }

        # Task Caption: 描述任务指令
        self.task_caption = {
    "instruction": "Place the cup in the middle drawer.",
    "num frame": "20",
    "obj_loc": {
      "cup": "(0.1, 0.2, 0.3)",
      "drawer": "(0.4, 0.5, 0.6)"
    },
    "task": "Task Caption",
    "example": [
      {
        "Q": "Initial scene <initial scene> and final scene <final scene>, ask what task the robot performed.",
        "A": "Place the cup in the middle drawer."
      }
    ]
  }

        # Embodied QA: 动态 3D 场景问答
        self.embodied_qa ={
    "instruction": "Place the cup in the middle drawer.",
    "num frame": "20",
    "obj_loc": {
      "cup": "(0.1, 0.2, 0.3)",
      "drawer": "(0.4, 0.5, 0.6)"
    },
    "task": "Embodied QA",
    "example": [
      {
        "Q": "Where was the cup initially?",
        "A": "The cup was initially located at (0.1, 0.2, 0.3).",
        "Q": "Where was the cup finally placed?",
        "A": "The cup was finally placed in the middle drawer at (0.4, 0.5, 0.6)."
      }
    ]
  }
        # Localization: 定位物体
        self.localization = {
    "instruction": "Place the cup in the middle drawer.",
    "num frame": "20",
    "obj_loc": {
      "cup": "(0.1, 0.2, 0.3)",
      "drawer": "(0.4, 0.5, 0.6)"
    },
    "task": "Localization",
    "example": [
      {
        "Q": "Detect where objects are, and answer the location of the objects.",
        "A": "The cup is located at (0.1, 0.2, 0.3), and the middle drawer is at (0.4, 0.5, 0.6)."
      }
    ]
  }

        # Dense Caption: 根据位置描述物体
        self.dense_caption = {
    "instruction": "Place the bowl in the middle drawer.",
    "num frame": "20",
    "obj_loc": {
      "bowl": "(0.1, 0.2, 0.3)",
      "middle drawer": "(0.4, 0.5, 0.6)"
    },
    "task": "Dense Caption",
    "example": [
      {
        "Q": "This (0.1, 0.2, 0.3) is?",
        "A": "A plastic, shiny and pink bowl in the middle drawer."
      }
    ]
  }

        # Image or Point Cloud Generation: 生成目标图像或点云
        self.image_or_point_cloud = {
    "instruction": "Pick up the chip bag.",
    "num frame": "20",
    "obj_loc": {
      "the chip bag": "(0.1, 0.2, 0.3)"
    },
    "task": "Image or Point Cloud Generation",
    "example": [
      {
        "Q": "The initial scene is <initial scene> Find some snacks for me.",
        "A": "Sure! I should <img> pick up <obj> the chip bag </obj>:(0.1, 0.2, 0.3)</img>"
      },
      {
        "Q": "The initial scene is <initial scene> Find some snacks for me.",
        "A": "Sure! I should <pcd> pick up <obj> the chip bag </obj>:(0.1, 0.2, 0.3)</pcd>"
      }
    ]
  }

        # Action Prediction: 预测机器人动作
        self.action_prediction = {
    "instruction": "Place the cup in the middle drawer.",
    "num frame": "20",
    "obj_loc": {
      "cup": "(0.1, 0.2, 0.3)",
      "middle drawer": "(0.4, 0.5, 0.6)"
    },
    "task": "Action Prediction",
    "example": [
      {
        "Q": "Given the <initial scene>, What should I do next?",
        "A": "By 3 steps: 1. Move the robotic arm to the cup's initial location at (0.1, 0.2, 0.3). \n2. Grasp the cup and move to the middle drawer at (0.4, 0.5, 0.6). \n3. Release the cup inside the middle drawer."
      },
      {
        "Q": "<initial scene> Place the cup in the middle drawer, execute now.",
        "A": "Actions are: <action>"
      }
    ]
  }

        # 将所有示例组合在一起
        self.samples = {
            "Verification": self.verification,
            "Task Caption": self.task_caption,
            "Embodied QA": self.embodied_qa,
            "Localization": self.localization,
            "Dense Caption": self.dense_caption,
            "Image or Point Cloud Generation": self.image_or_point_cloud,
            "Action Prediction": self.action_prediction
        }
        
    def generate_template_prompt(self,chouse_task):
        """
        根据任务信息生成ChatGPT的模板。 [toy car at (0.6, -0.4, 0.3), open box at (0.9, -0.5, 0.2)]
        参数:
            task_info (dict): 包含任务相关信息，如指令、场景状态、动作序列等
        返回:
            str: 生成的模板文本
        """
        task_prompt = f"""
                    You are an AI visual assistant and a question-answering generator capable of analyzing dynamic 3D scenes.\n
                    Suppose you have observed a robotic arm successfully executing an instruction: [instruction].\n
                    The scene's initial state is <initial scene> and <final scene>, where the final scene is the [num frame] frame, and we assume that the task was definitely not completed in the first 2/3 of the time.\n
                    You have the action sequence <action> of the robot arm.\n
                    In this instruction, the initial positions of these objects are [object + location]. Note that the location is the center points of objects represented by a 3D coordinate (x, y, z) with units of meters.\n
                    Utilizing all the information above, you can choose to rewrite the instruction while retaining its original meaning.\n
                    Further, you need to generate multiple rounds of dialogue or a question-answer pair, which should correspond to one of the following tasks:\n
                    1. Verification: Given the initial state and a mid-state frame, ask if the robot has completed the instruction.\n
                    2. Task Caption: Given the initial and final states, ask what task the robot performed.\n
                    3. Embodied QA: Please conduct some questions and answers about the current dynamic scene.\n
                    4. Localization: Detect where objects are, answer the location of the objects.\n
                    5. Dense Caption: Given the location of objects, answer with a description of those objects.\n
                    6. Image or Point Cloud Generation: Given the initial scene and instruction, generate an image or point cloud of the final state. If choosing this task, enclose the instruction with the <image> </image> or <pcd> </pcd> token to represent generation.\n
                    7. Action Prediction: Given the initial scene, or having both initial and final scenes, predict actions.\n
                    You can include a simple task decomposition, but the length of the decomposition must not exceed 3.\n
                    """
        prompt_question = (f"instruction:[{chouse_task['instruction']}]\n"
                    f"num frame:[{chouse_task['num frame']}]\n"
                    f"objects locations:[{format_object_locations(chouse_task['obj_loc'])}]\n"
                    f"choose task:{chouse_task['task']}\n"
                    )

        if len(chouse_task["example"])>1:
            example = random.choice(chouse_task["example"])
        else:
            example = chouse_task["example"][0]
        prompt_answer = f"Q:{example['Q']}\nA:{example['A']}"
        
        return task_prompt,prompt_question,prompt_answer

class InstructionTemplateV2:
    def __init__(self, initial_scene, current_scene=None, final_scene=None, instruction=None, obj=None, location=None, action=None, finished=None):
        self.initial_scene = initial_scene
        self.current_scene = current_scene
        self.final_scene = final_scene
        self.instruction = instruction
        self.obj = obj
        self.location = location
        self.action = action
        self.finished = finished

    def verification(self):
        question = f"The initial scene is <scene>{self.initial_scene}</scene> and the current scene is <scene>{self.current_scene}</scene>. Instruction: {self.instruction} Finished?"
        answer = f"{self.finished}"
        return question, answer

    def task_caption(self):
        question = f"The initial scene is <scene>{self.initial_scene}</scene> and the final scene is <scene>{self.final_scene}</scene>. Describe the task."
        answer = self.instruction
        return question, answer

    def localization(self):
        question = f"The scene is <scene>{self.initial_scene}</scene>. Locate: {self.obj}."
        answer = self.location
        return question, answer

    def dense_caption(self):
        question = f"The scene is <scene>{self.initial_scene}</scene>. What is located at {self.location}?"
        answer = self.obj
        return question, answer

    def image_generation(self):
        question = f"The initial scene is <scene>{self.initial_scene}</scene>. Instruction: {self.instruction} Generate the goal image."
        answer = f"<image> {self.instruction} </image>"
        return question, answer

    def point_cloud_generation(self):
        question = f"The initial scene is <scene>{self.initial_scene}</scene>. Instruction: {self.instruction} Generate the goal point cloud."
        answer = f"<pcd> {self.instruction} </pcd>"
        return question, answer

    def action_prediction(self):
        question = f"<scene>{self.initial_scene}</scene>. {self.instruction}. Predict actions."
        answer = self.action
        return question, answer
if __name__ == "__main__":

    # 输入示例
    initial_scene = "[initial_embed]"
    current_scene = "[current_embed]"
    final_scene = "[final_embed]"
    instruction = "Find some snacks for me."
    obj = "<obj> the chip bag </obj>"
    location = "[loc tokens]"
    action = "[action seq]"
    finished = "Yes/No"

    # 示例
    template_v2 = InstructionTemplateV2(
        initial_scene=initial_scene,
        current_scene=current_scene,
        final_scene=final_scene,
        instruction=instruction,
        obj=obj,
        location=location,
        action=action,
        finished=finished
    )

    # 打印示例
    print(template_v2.verification())
    print(template_v2.task_caption())
    
    # 示例1
    template_gpt = InstructionTemplateForGPT(
        initial_scene=initial_scene,
        current_scene=current_scene,
    )
    choose_task_name = random.choice(list(template_gpt.samples.keys()))
    choose_task = template_gpt.samples[choose_task_name]
    task_prompt, prompt_questions, prompt_answers = template_gpt.generate_template_prompt(choose_task)
    print(task_prompt)
    print(prompt_questions)
    print(prompt_answers)
