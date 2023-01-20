####################################
# Testing OpenAI's GPT-3
# for public services uses
# (this is a personal project
# unaffiliated with any actual
# public services or organisations)
####################################

import os
import openai
import pickle as pk

openai.api_key_path = "openai_api_key.txt"


def get_response(prompt: str, temp: int = 0.5, max_tokens: int = 2048):
    return openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=temp,
        max_tokens=max_tokens
    )


def get_edit_response(prompt: str, instruction: str, temp: int = 0.5, n: int = 20):
    return openai.Edit.create(
        engine="text-davinci-edit-001",
        input=prompt,
        instruction=instruction,
        temperature=temp,
        n=n
    )


def read_in_prompt_return_outputs(input_directory: str,
                                  output_directory: str,
                                  input_txt_file_name: str,
                                  complete_or_edit: str = "complete",
                                  **kwargs):
    root_name = input_txt_file_name.replace(".txt", "")
    file_containing_output = root_name + "_response.txt"
    pickle_object_dump_name = root_name + "_pickled_obj.pkl"

    with open(os.path.join(input_directory, input_txt_file_name), "r") as f:
        prompt = f.read()
        print(prompt)

    if complete_or_edit == "complete":
        model_response = get_response(prompt=prompt, **kwargs)
    elif complete_or_edit == "edit":
        model_response = get_edit_response(prompt=prompt, **kwargs)
    else:
        raise TypeError("complete_or_edit should be 'complete' or 'edit'")

    with open(os.path.join(output_directory, file_containing_output), "w") as f:
        f.write(model_response.choices[0].text)
    with open(os.path.join(output_directory, pickle_object_dump_name), "wb") as f:
        pk.dump(model_response, f)

    print(f"All done, files save in {output_directory}")
    return model_response


if __name__ == "__main__":
    mod_response = read_in_prompt_return_outputs(input_directory="prompt_material",
                                                 output_directory="gpt_output",
                                                 input_txt_file_name="single_patient_history_piecewise_combined_prompt_response_edited.txt",
                                                 temp=0.8)

    edit_response = read_in_prompt_return_outputs(input_directory="prompt_material",
                                                  output_directory="gpt_output",
                                                  input_txt_file_name="__.txt",
                                                  temp=0.8,
                                                  complete_or_edit="edit",
                                                  instruction="Add in some mistakes and irrelevant details.",
                                                  n=20)

