class Person:
    def __init__(self, name, age=23):
        self.name = name
        self.age = age
    
    def __str__(self):
        return f"{self.age}-aged person named {self.name}"

class Dog:
    pass


p1 = Person("Abby")
d1 = Dog()
d1.name = "terrier"
d1.age = 7
d1.weight_kg = 30

print(p1.name)
print(p1.age)
print()
print(d1.name)
print(d1.age)
print(d1.weight_kg)

print(p1)