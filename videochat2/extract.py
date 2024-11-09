import json
import os


# git clone https://huggingface.co/datasets/OpenGVLab/VideoChat2-IT

def read_json(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data


def save_json(path, questions):
    with open(path, 'w') as output_file:
        json.dump(questions, output_file, indent=4)
    print(f"Questions extracted and saved to '{path}'")


def extract_many(root_dir):
    structured_data = {}

    for task in os.listdir(root_dir):
        task_path = os.path.join(root_dir, task)
        if os.path.isdir(task_path):
            structured_data[task] = {}  # Initialize task level
            for dataset in os.listdir(task_path):
                dataset_path = os.path.join(task_path, dataset)
                if os.path.isdir(dataset_path):
                    # Initialize dataset level in the structure
                    structured_data[task][dataset] = []

                    instruction_file_path = os.path.join(dataset_path, 'instructions.json')
                    # print(dataset_path)
                    if os.path.exists(instruction_file_path):
                        with open(instruction_file_path, 'r') as file:
                            data = json.load(file)
                            if isinstance(data, list):
                                structured_data[task][dataset].extend(data)
                    elif dataset_path == r"video\grounding\didemo":
                        instruction_file_path_1 = os.path.join(dataset_path, 'instructions_a2t.json')
                        instruction_file_path_2 = os.path.join(dataset_path, 'instructions_t2a.json')
                        with open(instruction_file_path_1, 'r') as file:
                            data = json.load(file)
                            if isinstance(data, list):
                                structured_data[task][dataset].extend(data)
                        with open(instruction_file_path_2, 'r') as file:
                            data = json.load(file)
                            if isinstance(data, list):
                                structured_data[task][dataset].extend(data)
                    else:
                        for json_file in os.listdir(dataset_path):
                            if json_file.endswith('.json'):
                                file_path = os.path.join(dataset_path, json_file)
                                with open(file_path, 'r') as file:
                                    data = json.load(file)
                                    if isinstance(data, list):
                                        instructions = [item['QA'][0]['q'] for item in data]
                                        structured_data[task][dataset].extend(instructions)
    save_json(f"{root_dir}_instruction.json", structured_data)


def extract_text(path):
    data = read_json(path)
    # for item in data:
    #     print(item['QA'][0]['q'])
    questions = [{'q': item['QA'][0]['q']} for item in data]
    save_json("text_instruction.json", questions)


def main():
    path_text = "text/sharegpt/train.json"
    path_img = "image"
    path_video = "video"

    extract_many(path_img)
    extract_many(path_video)
    extract_text(path_text)


if __name__ == '__main__':
    main()
