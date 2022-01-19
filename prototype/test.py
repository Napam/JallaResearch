class Person:
    def __init__(self, firstName, lastName, age) -> None:
        self.firstName = firstName 
        self.lastName = lastName
        self.age = age
        print("Created person " + firstName)

Person.greet = lambda self: print(f"Hello I am {self.firstName}")

class Baby(Person):
    def __init__(self, firstName, lastName, age) -> None:
        super().__init__(firstName, lastName, age)
    
Baby.greet = lambda self: print(f"Baaaa {self.firstName}")
Baby.__mro__ = list(Baby.__mro__)

a = Person("Naphat", "Amundsen", 24)
a.greet()
print(Baby.__mro__)