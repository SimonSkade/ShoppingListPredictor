file = open("0.txt", "r") 
lines = file.readlines()

features = (sorted(lines[0].replace(" ", "").upper().split(",")))

vectors = []
print(features)
for l in range(len(lines)-1):
	vectors.append(lines[l+1].upper().split(","))

for i, v in enumerate(vectors[0]):
	if int(v)==1: print(features[i])
