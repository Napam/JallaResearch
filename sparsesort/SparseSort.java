import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;
import java.util.stream.IntStream;


class SparseSort {
    public static void main(String[] args) {
        // simple();
        graph1();
        System.out.println();
        graph2();
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
    
    public static void graph1() {
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
        Map<String, Integer> counter = new HashMap<>();
        for (Set<String> descendants : graph.values()) {
            for (String descendant : descendants) {
                counter.putIfAbsent(descendant, 0);
                counter.compute(descendant, (k, v) -> v + 1);
            }
        }
        System.out.println(graph);
        System.out.println(counter);
        sort(new ArrayList<>(nodes.keySet()), counter);
    }
    
    public static void graph2() {
        LinkedHashMap<String, List<String>> graph = new LinkedHashMap<>();
        graph.put("A", Arrays.asList("B"));
        graph.put("B", Arrays.asList("C", "E"));
        graph.put("C", Arrays.asList());
        graph.put("D", Arrays.asList("E"));
        graph.put("E", Arrays.asList("C"));
        graph.put("F", Arrays.asList("D"));
        graph.put("G", Arrays.asList());
        graph.put("H", Arrays.asList("I"));
        graph.put("I", Arrays.asList("J"));
        graph.put("J", Arrays.asList("H"));
        graph = reverseEdges(graph);
        Map<String, Integer> counts = countDescendantsGraph(graph);
        System.out.println(graph);
        System.out.println(counts);
        sort(new ArrayList<>(graph.keySet()), counts);
    }

    public static <T> LinkedHashMap<T, List<T>> reverseEdges(Map<T, List<T>> graph) {
        LinkedHashMap<T, List<T>> newGraph =  new LinkedHashMap<>();
        for (Entry<T, List<T>> entry : graph.entrySet()) {
            T parent = entry.getKey();
            newGraph.putIfAbsent(parent, new ArrayList<>());
            for (T child : entry.getValue()) {
                List<T> list = newGraph.getOrDefault(child, new ArrayList<>());
                list.add(parent);
                newGraph.put(child, list);
            }
        }
        return newGraph;
    }

    public static <T> Integer countDescendants(
        T curr, 
        Map<T, List<T>> nodes,
        Map<T, Integer> graph
    ) {
        Integer descendantsOfCurr = graph.get(curr);
        if (descendantsOfCurr != null)
            return descendantsOfCurr;

        graph.put(curr, 0);
        for (T child : nodes.get(curr))
            graph.compute(curr, (k, v) -> v + 1 + countDescendants(child, nodes, graph));
        return graph.get(curr);
    }

    public static <T> Map<T, Integer> countDescendantsGraph(Map<T, List<T>> graph) {
        LinkedHashMap<T, Integer> counts = new LinkedHashMap<>();
        graph.entrySet().forEach(entry -> countDescendants(entry.getKey(), graph, counts));    
        return counts;
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