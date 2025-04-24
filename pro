
EXP 2
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27,16,2);

#define ledred 12
#define ledorange 13
#define ledgreen 14

void setup(){
  Serial.begin(115200);
  pinMode(ledred,OUTPUT);
  pinMode(ledorange,OUTPUT);
  pinMode(ledgreen,OUTPUT);
}
void loop(){
  digitalWrite(ledred,HIGH);
  digitalWrite(ledorange,LOW);
  digitalWrite(ledgreen,LOW);
  delay(3000);

  digitalWrite(ledred,LOW);
  digitalWrite(ledorange,HIGH);
  digitalWrite(ledgreen,LOW);
  delay(3000);

  digitalWrite(ledred,LOW);
  digitalWrite(ledorange,LOW);
  digitalWrite(ledgreen,HIGH);
  delay(3000);
}


EXP 3
#define led1 14
#define led2 12
#define IR_Sensor 16

void setup(){
  Serial.begin(9600);
  pinMode(led1,OUTPUT);
  pinMode(led2,OUTPUT);
  pinMode(IR_Sensor,INPUT);
}

void loop(){
  if(digitalRead(IR_Sensor) == HIGH){
    digitalWrite(led1,LOW);
    digitalWrite(led2,LOW);
    delay(500);
  }
  else {
    digitalWrite(led1,HIGH);
    digitalWrite(led2,HIGH);
    delay(500);

  }
}


EXP 4
#define LDRsensor A0

void setup(){
  Serial.begin(9600);
  pinMode(LDRsensor,INPUT);
}

void loop(){
  int sensor_value = analogRead(LDRsensor);
  int light_value = map(sensor_value,0,1023,100,0);
  Serial.print("Intensity value : (%) ");
  Serial.print(light_value);
  delay(500);
}

EXP 5
#define led1 5
#define led2 4
#define pushbutton 14

void setup(){
  pinMode(pushbutton,INPUT);
  pinMode(pushbutton,OUTPUT);
  pinMode(pushbutton,OUTPUT);
}

void loop(){
  if(digitalRead(pushbutton) == LOW){
    digitalWrite(led1,HIGH);
    digitalWrite(led2,LOW);
  }
  else {
    digitalWrite(led1,LOW);
    digitalWrite(led2,HIGH);
  }
}


EXP 6
#include <DHT.h>
#include <LiquidCrystal_I2C.h>
#include <Wire.h>

LiquidCrystal_I2C lcd(0x27,16,2);

#define DHT_pin 14

#define DHTTYPE DHT11

DHT dht(DHT_pin,DHTTYPE);

float temperature_value, humidity_value;

void setup(){
  lcd.begin();
  lcd.setCursor(0,0);
  lcd.print("Temperature");
  lcd.setCursor(0,1);
  lcd.print("Monitoring System");
  delay(500);
  pinMode(DHT_pin,INPUT);
  dht.begin();
}

void loop(){
  humidity_value = dht.readHumidity();
  temperature_value = dht.readTemperature();

  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print("Temperature :");
  lcd.setCursor(0,1);
  lcd.print(temperature_value);
  lcd.print("*c");
  delay(500);

  lcd.clear();
  lcd.setCursor(0,0);
  lcd.print("Humidity : ");
  lcd.setCursor(0,1);
  lcd.print(humidity_value);
  lcd.print('%');
  delay(500);

}


EXP 7

#define led1 4
#define led2 5
#define buzzer 14
char input;
void setup(){
  Serial.begin(9600);
  pinMode(led1,OUTPUT);
  pinMode(led2,OUTPUT);
  pinMode(buzzer,OUTPUT);
}

void loop(){
  if(Serial.available()>0){
    input = Serial.read();
    Serial.print(input);
    if(input == 'A'){
      Serial.println("A Received");
      digitalWrite(led1,HIGH);
      delay(250);
    }
    else if(input == 'a'){
      Serial.println("a Received");
      digitalWrite(led2,LOW);
      delay(250);
    }
    if(input == 'B'){
      Serial.println("B Received");
      digitalWrite(led2,HIGH);
      delay(250);
    }
    else if(input == 'b'){
      Serial.println("b Received");
      digitalWrite(led2,LOW);
      delay(250);
    }
    if(input == 'C'){
      Serial.println("C Received");
      digitalWrite(buzzer,HIGH);
      delay(250);
    }
    else if(input == 'c'){
      Serial.println("c Received");
      digitalWrite(buzzer,LOW);
      delay(250);
    }
    input = '\n';
  }
}

EXP 8

#include "Ubidots.h"

const char* UBIDOTSTOKEN = "BBUS-nJ4ubqCMhdap26u7PdI283CNaDDQGv";
const char* SSID = "Mohammad Rafi";
const char* PASS = "@Mohammad18169";

Ubidots ubidots(UBIDOTSTOKEN,UBI_HTTP);
#define temp_sensor A0

float celsius,fahrenheit;

void setup(){
  Serial.begin(115200);
  ubidots.wifiConnect(SSID,PASS);

  pinMode(temp_sensor,INPUT);
  delay(500);
}

void loop(){
  celsius = (analogRead(temp_sensor)*330.0)/1023.0;
  fahrenheit = celsius*1.8 + 32.0;
  ubidots.add("Celsius", celsius);
  ubidots.add("Fahrenheit",fahrenheit);
  ubidots.send();
  delay(1000);

}


EXP 9

const int relaypin = D1;
const int ldrpin = A0;

void setup(){
  Serial.begin(9600);
  pinMode(relaypin,OUTPUT);
  pinMode(ldrpin,INPUT);
  delay(1000);
}
void loop(){
  if(analogRead(ldrpin)<500){
    digitalWrite(relaypin,HIGH);
    Serial.println("dark output . Switch ON");
  }
  else {
    digitalWrite(relaypin,LOW);
    Serial.println("Bright outside. Switch OFF");
    
  }
  delay(1000);
}
