import java.io.*;
public class FileFinder{
	public static void main(String args[]){
		File folder = new File("F:\\Rafi\\My_Study\\4_1\\AI_Lab\\aclImdb\\test");
		File[] listOfFiles = folder.listFiles();
		System.out.println(listOfFiles.length);
		for (int i = 0;i<listOfFiles.length;++i){
			System.out.println(listOfFiles[i].getName());
		}
    }
}