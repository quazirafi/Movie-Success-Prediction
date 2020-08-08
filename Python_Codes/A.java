import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.concurrent.TimeUnit;

public class A {
    
    public String execute(){
         Process p;
try{
    System.out.println("SEND");
    String cmd = "python tokenizer3.py";
    //System.out.println(cmd);
    p = Runtime.getRuntime().exec(cmd); 
    BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()));
    String s = br.readLine(); 
    System.out.println(s);
    System.out.println("Sent");
    p.wait(180000);
    p.destroy();
} catch (Exception e) {}
        return "successfully";
    }
    
    public static void main(String args[]){
        A a = new A();
        a.execute();
    }
       
}
