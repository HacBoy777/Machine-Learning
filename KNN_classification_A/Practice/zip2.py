price = [10,25,30,15,5]
item = ["Burger","Rolls","Pizza","Pasta","Sandwich"]

# x = zip(price,item)
# # print(list(x))

# money,food = zip(*x)

# money,food = zip(*sorted(zip(price,item)))
money,food = (list(t) for t in zip(*sorted(zip(price,item))))
print("Money : ",money)
print("Food : ",food)