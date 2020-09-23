import re

with open("helper_gemuese.txt","r") as f:
	file = f.read()
	products = re.findall("</a>\s*[A-Z][a-z,ü,ö,ä,ß]*\s*[A-Z,a-z,(,)]*<br />", file)
	prods = []
	for p in products:
		prods.append(re.findall(r"[A-Z][a-z]*", p)[0])
	res = ""
	for p in prods:
		res += p + "\n"
	print(res)