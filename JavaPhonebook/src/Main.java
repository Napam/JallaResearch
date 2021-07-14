import java.util.Scanner;

public class Main {
    public static void main (String[] args)
    {
        Phonebook book;
        Scanner sc;
        int choice;

        book = new Phonebook();
        book.add("Naphat", "Amundsen", 48424120);
        book.add("Jennifer", "Mjøs", 40447743);
        book.add("Narkesha", "Alfanta", 45519033);

        book.printEntries();
    }
}
