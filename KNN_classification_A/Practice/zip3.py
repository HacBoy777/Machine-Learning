names = ["Ravi", "Anita", "Karan", "Meena"]
marks = [78, 92, 85, 88]

# x = zip(names, marks)
# # print(list(x))

sorted_names, sorted_marks = zip(*sorted(zip(names, marks)))
print("Sorted Names : ",sorted_names)
print("Sorted Marks : ",sorted_marks)
