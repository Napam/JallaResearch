import java.util.Scanner;

public class Main {
    private static void printMenu()
    {
        System.out.println("    Phonebook");
        System.out.println("=================");
        System.out.println("0. Exit");
        System.out.println("1. Print entries");
        System.out.println("2. Add entry");
        System.out.println("3. Remove entry");
    }

    public static void main (String[] args)
    {
        Phonebook book;
        Scanner sc;
        int choice;

        book = new Phonebook();
        book.add("Naphat", "Amundsen", 48424120);
        book.add("Jennifer", "Mjøs", 40447743);
        book.add("Narkesha", "Alfanta", 45519033);

        sc = new Scanner(System.in);

        scanner: while (true)
        {
            printMenu();
            choice = sc.nextInt();

            switch (choice) {
                case 0:
                    break scanner;
                case 1: // Print entries
                    System.out.println();
                    book.printEntries();
                    System.out.println();
                    break;
                case 2: // Add entry
                    book.cli_add();
                    break;
                case 3: // Remove entry
                    book.cli_remove();
                    break;
                default:
                    break;
            }
        }
    }
}
