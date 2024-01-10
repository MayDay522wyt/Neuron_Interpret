def make_explanation_prompt(all_activation_records):
    prompt = ("We're studying neurons in a neural network. Each neuron looks for some particular "
            "thing in a short document. Look at the parts of the document the neuron activates for "
            "and summarize in a single sentence what the neuron is looking for. Don't list "
            "examples of words.\n\nThe activation format is concept<tab>activation. Activation "
            "values range from 0 to 1. A neuron finding what it's looking for is represented by a "
            "non-zero activation value. The higher the activation value, the stronger the match.")

    prompt += "\n\n"

    for i, activation_records in enumerate(all_activation_records):
        prompt += f"\nExample {i+1}:\n"
        for key in activation_records.keys():
            prompt += f"{key}\t{activation_records[key]}\n"
    

    return prompt