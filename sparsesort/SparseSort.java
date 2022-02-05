import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;


class SparseSort {
    public static void main(String[] args) {
        System.out.println("simple");
        simple();
        System.out.println();
        System.out.println("graph1");
        graph1();
        System.out.println();
        System.out.println("graph2");
        graph2();
        System.out.println();
        System.out.println("graph3");
        graph3();
    }

    public static <T> Set<T> getDescendants(
        T curr, 
        Map<T, List<T>> children,
        Map<T, Set<T>> descendants
    ) {
        Set<T> descendantsOfCurr = descendants.get(curr);
        if (descendantsOfCurr != null)
            return descendantsOfCurr;

        descendantsOfCurr = new HashSet<>();
        descendants.put(curr, descendantsOfCurr);
        for (T child : children.get(curr)) {
            descendantsOfCurr.add(child);
            descendantsOfCurr.addAll(getDescendants(child, children, descendants));
        }
        
        return descendantsOfCurr;
    }

    public static <T> Map<T, Set<T>> getDescendantsGraph(Map<T, List<T>> children) {
        LinkedHashMap<T, Set<T>> descendants = new LinkedHashMap<>();
        children.entrySet().forEach(entry -> getDescendants(entry.getKey(), children, descendants));    
        return descendants;
    }

    public static LinkedHashMap<String, List<String>> children = new LinkedHashMap<>();

    static {
        children.put("A", Arrays.asList("B"));
        children.put("B", Arrays.asList("C", "E"));
        children.put("C", Arrays.asList());
        children.put("D", Arrays.asList("E"));
        children.put("E", Arrays.asList("C"));
        children.put("F", Arrays.asList("D"));
        children.put("G", Arrays.asList());
        children.put("H", Arrays.asList("I"));
        children.put("I", Arrays.asList("J", "D"));
        children.put("J", Arrays.asList("H"));
    }

    
    public static void graph1() {
        Map<String, Set<String>> children_ = getDescendantsGraph(children);
        Map<String, Integer> counter = new HashMap<>();
        for (Set<String> descendants : children_.values()) {
            for (String descendant : descendants) {
                counter.putIfAbsent(descendant, 0);
                counter.compute(descendant, (k, v) -> v + 1);
            }
        }
        System.out.println(children_);
        System.out.println(counter);
        sort(new ArrayList<>(children_.keySet()), counter);
    }
    
    public static void graph2() {
        Map<String, List<String>> children_ = reverseEdges(children);
        Map<String, Integer> counts = countDescendantsGraph(children_);
        System.out.println(children_);
        System.out.println(counts);
        sort(new ArrayList<>(children_.keySet()), counts);
    }
    
    public static void graph3() {
        Map<String, List<String>> children_ = reverseEdges(children);
        Map<String, Set<String>> parentTable = getDescendantsGraph(children_);
        Map<String, Integer> counts = new HashMap<>();
        parentTable.forEach((node, parents) -> counts.put(node, parents.size()));
        System.out.println(parentTable);
        System.out.println(counts);
        sort(new ArrayList<>(children_.keySet()), counts);
    }

    public static <T> Map<T, List<T>> reverseEdges(Map<T, List<T>> graph) {
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
        List<String> strings = Stream.concat(children.keySet().stream(), children.keySet().stream()).collect(Collectors.toList());
        List<String> ordering = Arrays.asList(
            "C",
            "A",
            "G",
            "E",
            "D"
        );
        HashMap<String, Integer> map = new HashMap<>();
        IntStream.range(0, ordering.size()).forEach(i -> map.put(ordering.get(i), i));
        sort(strings, map);
    }
    
    public static void sort(List<String> strings, Map<String, Integer> priority) {
        strings.sort((right, left) -> priority.getOrDefault(right, 0) - priority.getOrDefault(left, 0));
        System.out.println(strings);
    }
}