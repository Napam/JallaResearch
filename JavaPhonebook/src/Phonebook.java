import java.util.ArrayList;
import java.util.Scanner;

public class Phonebook {
    int longestName; // Length og "Ola Nordmann"
    ArrayList<Entry> entries;

    public Phonebook ()
    {
        entries = new ArrayList<>();
        longestName = -1;
    }

    public boolean add(Entry e)
    {
        int nameLen = e.firstName.length() + e.lastName.length() + 11;

        // For printing purposes
        if (longestName < nameLen) {
            longestName = nameLen;
        }

        return entries.add(e);
    }

    public boolean add(String firstName, String lastName, int number)
    {
        Entry e = new Entry(firstName, lastName, number);
        return this.add(e);
    }

    public void remove(String firstName, String lastName)
    {
        for (Entry e : entries) {
            if (e.firstName.equals(firstName) && e.lastName.equals(lastName))
            {
                entries.remove(e);
                break;
            }
        }
    }

    public void cli_remove()
    {
        Scanner sc;
        String firstName, lastName;
        sc = new Scanner(System.in);
        System.out.print("Enter first name: ");
        firstName = sc.next();
        System.out.print("Enter last name: ");
        lastName = sc.next();

        remove(firstName, lastName);
    }

    public boolean cli_add()
    {
        Scanner sc;
        String firstName, lastName;
        Entry e;
        int number;

        sc = new Scanner(System.in);

        System.out.print("Enter first name: ");
        firstName = sc.next();
        System.out.print("Enter last name: ");
        lastName = sc.next();
        System.out.print("Enter number: ");
        number = sc.nextInt();

        e = new Entry(firstName, lastName, number);
        System.out.printf("Added new entry:\n%s", e);

        return this.add(e);
    }

    public void printEntries()
    {
        for (Entry e : entries) {
            System.out.printf("%" + longestName + "s%n", e);
        }
    }
}
