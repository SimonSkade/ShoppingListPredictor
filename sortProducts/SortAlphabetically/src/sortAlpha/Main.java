package sortAlpha;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;

public class Main {
	static List<String> lines;

	public static void main(String[] args) throws IOException {
		try {
			lines = Files.readAllLines(Paths.get("C:/Users/Prog/Desktop/GroceryListPredictor/webapp/allproducts2.txt"));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		Collections.sort(lines);
		System.out.println(lines);
		FileWriter writer = new FileWriter("C:/Users/Prog/Desktop/GroceryListPredictor/webapp/sortedProducts.txt");
		for (String str : lines) {
			writer.write(str + System.lineSeparator());
		}
		writer.close();
	}
}
