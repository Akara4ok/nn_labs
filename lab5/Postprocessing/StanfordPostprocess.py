def postprocess_label(label):
    name_start = label.find('-') + 1
    return label[name_start:].replace('_', ' ').capitalize()