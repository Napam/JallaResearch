import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;
import java.util.stream.Collector;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


class SparseSort {
    public static void main(String[] args) {
        // simple();
        graph();
    }

    public static <T> Set<T> getDescendants(
        T curr, 
        Map<T, List<T>> nodes,
        Map<T, Set<T>> graph
    ) {
        Set<T> descendantsOfCurr = graph.get(curr);
        if (descendantsOfCurr != null)
            return descendantsOfCurr;

        descendantsOfCurr = new HashSet<>();
        graph.put(curr, descendantsOfCurr);
        for (T child : nodes.get(curr)) {
            descendantsOfCurr.add(child);
            descendantsOfCurr.addAll(getDescendants(child, nodes, graph));
        }

        return descendantsOfCurr;
    }

    public static <T> Map<T, Set<T>> getDescendantsGraph(Map<T, List<T>> nodes) {
        LinkedHashMap<T, Set<T>> graph = new LinkedHashMap<>();
        nodes.entrySet().forEach(entry -> getDescendants(entry.getKey(), nodes, graph));    
        return graph;
    }

    public static void graph() {
        LinkedHashMap<String, List<String>> nodes = new LinkedHashMap<>();
        nodes.put("A", Arrays.asList("B"));
        nodes.put("B", Arrays.asList("C", "E"));
        nodes.put("C", Arrays.asList());
        nodes.put("D", Arrays.asList("E"));
        nodes.put("E", Arrays.asList("C"));
        nodes.put("F", Arrays.asList("D"));
        nodes.put("G", Arrays.asList());
        nodes.put("H", Arrays.asList("I"));
        nodes.put("I", Arrays.asList("J"));
        nodes.put("J", Arrays.asList("H"));
        Map<String, Set<String>> graph = getDescendantsGraph(nodes);
        System.out.println(graph);
        // sort(nodes.keySet().stream().collect(Collectors.toList()), dependencyGraph);
    }

    public static void simple() {
        List<String> strings = Arrays.asList(
            "A",
            "B",
            "C",
            "D",
            "E",
            "F"
        );
    
        List<String> ordering = Arrays.asList(
            "B",
            "A"
        );
    
        HashMap<String, Integer> map = new HashMap<>();
        IntStream.range(0, ordering.size()).forEach(i -> map.put(ordering.get(i), i));
        sort(strings, map);
    }
    
    public static void sort(List<String> strings, Map<String, Integer> precedenceTable) {
        strings.sort((l, r) -> precedenceTable.getOrDefault(l, -1) - precedenceTable.getOrDefault(r, -1) );
        System.out.println(strings);
    }
}