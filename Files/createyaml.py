yaml_content = """path: objdetection/Files/images/YoloDataset
train: images/train
val: images/val

nc: 17
names:
  - Angelina Jolie
  - Jennifer Lawrence
  - Megan Fox
  - Sandra Bullock
  - Will Smith
  - Brad Pitt
  - Johnny Depp
  - Natalie Portman
  - Scarlett Johansson
  - Denzel Washington
  - Kate Winslet
  - Nicole Kidman
  - Tom Cruise
  - Hugh Jackman
  - Leonardo DiCaprio
  - Robert Downey Jr
  - Tom Hanks
"""

with open("data.yaml", "w") as f:
    f.write(yaml_content)

print("✅ data.yaml created!")