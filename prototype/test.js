function Person(firstName, lastName, age) {
    this.firstName = firstName
    this.lastName = lastName
    this.age = age
    console.log("Created person " + firstName)
}

Person.bing = "bong"

Person.prototype.greet = function () {
    console.log(`Hello I am ${this.firstName}`)
}

function Baby(firstName, lastName, age) {
    Person.call(this, firstName, lastName, age)
}

Baby.prototype.greet = function() {
    console.log('Baaaaaaa ' + this.firstName)
}

let a = new Baby("Naphat", "Amundsen", 24)
console.log(Person.bing)