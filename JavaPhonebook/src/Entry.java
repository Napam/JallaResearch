public class Entry {
    String firstName;
    String lastName;
    int number;

    public Entry(String firstName, String lastName, int number)
    {
        this.firstName = firstName;
        this.lastName = lastName;
        this.number = number;
    }

    public String toString()
    {
        return this.firstName + " " + this.lastName + ", " + String.format("%8d", number);
    }
}
