import glob
import json

myFilesPaths = glob.glob(r'./dataset/Intents/\*.txt')
response_tag = '<Response>'
json_main ={"intents":[]}
sentences = []
for path in myFilesPaths:
    
    f = open(path, "r",encoding="utf8")
    lines =f.read().strip().replace('\t', '').split('\n')
    lines[:] = [x for x in lines if x]

    data = lines[8:]
    intent = data[0].replace(' ','').replace('(','').replace(')','').replace(':','').replace('\'','').lower()
    data = data[2:]
    try:
        response_index = data.index(response_tag)
    except:
        print(lines)
    

    responses = data[response_index+1:]
    patterns = data[:response_index]
    
    patterns_lower = []
    for pattern in patterns:
        patterns_lower.append(pattern.lower())
    
    for pattern_lower in patterns_lower:
        if pattern_lower in sentences:
            patterns_lower.remove(pattern_lower)
            print(pattern_lower)
        else:
            sentences.append(pattern_lower)
    
    # prepare JSON:

    json_object = {
    "tag": intent,
    "patterns": patterns_lower,
    "responses": responses
    }
    json_main["intents"].append(json_object)
#write json to file
with open('myIntents.json', 'w') as outfile:
    json.dump(json_main, outfile)

#print('intent:',intent,'\n\ndata:', data, f'\n\npatterns({patterns.__len__()}):', patterns, f'\n\nresponses({responses.__len__()}):', responses)
print('File was successfully created.')
