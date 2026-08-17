import numpy as np

products = ["Pen", "Book", "Bag", "Pencil"]
prices = [20, 120, 450, 10]

sorted_name, sorted_marks = ( list(t) for t in zip(*sorted(zip(products, prices), key=lambda x: x[1])))

print("sortedname :", sorted_name)
print("sortedmarks :", sorted_marks)