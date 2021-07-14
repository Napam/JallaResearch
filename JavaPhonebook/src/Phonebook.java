import java.util.ArrayList;

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

    public void printEntries()
    {
        for (Entry e : entries) {
            System.out.printf("%" + longestName + "s%n", e);
        }
    }
}
