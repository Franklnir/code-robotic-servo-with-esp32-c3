#include <WiFi.h>
#include <WiFiClientSecure.h> // Explicitly include for WiFiClientSecure
#include <PubSubClient.h>     // For MQTT Client
#include <ESP32Servo.h>       // For Servo control

// =========================================================
//                   WI-FI & MQTT CONFIGURATION
// =========================================================
const char* ssid = "GEORGIA";         // <<<< GANTI DENGAN NAMA WI-FI RUMAH ANDA
const char* password = "Georgia12345"; // <<<< GANTI DENGAN PASSWORD WI-FI RUMAH ANDA

// HiveMQ MQTT Broker Details
const char* mqtt_server = "7031e8491d334b88b475025af999ffeb.s1.eu.hivemq.cloud"; // From your image
const int mqtt_port = 8883; // TLS MQTT port from your image
const char* mqtt_username = "esp32cam"; // <<<< GANTI DENGAN USERNAME HIVEMQ ANDA
const char* mqtt_password = "Bracketservo2"; // <<<< GANTI DENGAN PASSWORD HIVEMQ ANDA

// MQTT Topics
#define MQTT_TOPIC_PAN_CMD "esp32/commands/pan"
#define MQTT_TOPIC_TILT_CMD "esp32/commands/tilt"
#define MQTT_TOPIC_PAN_STATUS "esp32/status/pan"
#define MQTT_TOPIC_TILT_STATUS "esp32/status/tilt"

// Certificates for TLS connection (HiveMQ requires this for 8883)
// You need to get the CA certificate from HiveMQ.
// Go to your HiveMQ Cloud cluster -> TLS Details -> Download CA Certificate.
// Open the .pem file with a text editor and copy its content here.
// NOTE: The certificate you provided is ISRG Root X1. While HiveMQ might use Let's Encrypt,
// it's generally best to use the specific CA certificate provided by HiveMQ for your cluster.
// If you encounter connection issues, try downloading the CA certificate directly from HiveMQ Cloud.
const char* hivemq_ca_cert = R"EOF(
-----BEGIN CERTIFICATE-----
MIIFazCCA1OgAwIBAgIRAIIQz7DSQONZRGPgu2OCiwAwDQYJKoZIhvcNAQELBQAw
TzELMAkGA1UEBhMCVVMxKTAnBgNVBAoTIEludGVybmV0IFNlY3VyaXR5IFJlc2Vh
cmNoIEdyb3VwMRUwEwYDVQQDEwxJU1JHIFJvb3QgWDEwHhcNMTUwNjA0MTEwNDM4
WhcNMzUwNjA0MTEwNDM4WjBPMQswCQYDVQQGEwJVUzEpMCcGA1UEChMgSW50ZXJu
ZXQgU2VjdXJpdHkgUmVzZWFyY2ggR3JvdXAxFTATBgNVBAMTDElTUkcgUm9vdCBY
MTCCAiIwDQYJKoZIhvcNAQEBBQADggIPADCCAgoCggIBAK3oJHP0FDfzm54rVygc
h77ct984kIxuPOZXoHj3dcKi/vVqbvYATyjb3miGbESTtrFj/RQSa78f0uoxmyF+
0TM8ukj13Xnfs7j/EvEhmkvBioZxaUpmZmyPfjxwv60pIgbz5MDmgK7iS4+3mX6U
A5/TR5d8mUgjU+g4rk8Kb4Mu0UlXjIB0ttov0DiNewNwIRt18jA8+o+u3dpjq+sW
T8KOEUt+zwvo/7V3LvSye0rgTBIlDHCNAymg4VMk7BPZ7hm/ELNKjD+Jo2FR3qyH
B5T0Y3HsLuJvW5iB4YlcNHlsdu87kGJ55tukmi8mxdAQ4Q7e2RCOFvu396j3x+UC
B5iPNgiV5+I3lg02dZ77DnKxHZu8A/lJBdiB3QW0KtZB6awBdpUKD9jf1b0SHzUv
KBds0pjBqAlkd25HN7rOrFleaJ1/ctaJxQZBKT5ZPt0m9STJEadao0xAH0ahmbWn
OlFuhjuefXKnEgV4We0+UXgVCwOPjdAvBbI+e0ocS3MFEvzG6uBQE3xDk3SzynTn
jh8BCNAw1FtxNrQHusEwMFxIt4I7mKZ9YIqioymCzLq9gwQbooMDQaHWBfEbwrbw
qHyGO0aoSCqI3Haadr8faqU9GY/rOPNk3sgrDQoo//fb4hVC1CLQJ13hef4Y53CI
rU7m2Ys6xt0nUW7/vGT1M0NPAgMBAAGjQjBAMA4GA1UdDwEB/wQEAwIBBjAPBgNV
HRMBAf8EBTADAQH/MB0GA1UdDgQWBBR5tFnme7bl5AFzgAiIyBpY9umbbjANBgkq
hkiG9w0BAQsFAAOCAgEAVR9YqbyyqFDQDLHYGmkgJykIrGF1XIpu+ILlaS/V9lZL
ubhzEFnTIZd+50xx+7LSYK05qAvqFyFWhfFQDlnrzuBZ6brJFe+GnY+EgPbk6ZGQ
3BebYhtF8GaV0nxvwuo77x/Py9auJ/GpsMiu/X1+mvoiBOv/2X/qkSsisRcOj/KK
NFtY2PwByVS5uCbMiogziUwthDyC3+6WVwW6LLv3xLfHTjuCvjHIInNzktHCgKQ5
ORAzI4JMPJ+GslWYHb4phowim57iaztXOoJwTdwJx4nLCgdNbOhdjsnvzqvHu7Ur
TkXWStAmzOVyyghqpZXjFaH3pO3JLF+l+/+sKAIuvtd7u+Nxe5AW0wdeRlN8NwdC
jNPElpzVmbUq4JUagEiuTDkHzsxHpFKVK7q4+63SM1N95R1NbdWhscdCb+ZAJzVc
oyi3B43njTOQ5yOf+1CceWxG1bQVs5ZufpsMljq4Ui0/1lvh+wjChP4kqKOJ2qxq
4RgqsahDYVvTH9w7jNbyLeiNdd8XM2w9U/t7y0Ff/9yi0GE44Za4rF2LN9d11TPA
mRGunUHBcnWEvgJBQl9nJEiU0Zsnvgc/ubhPgXRR4Xq37Z0j4r7g1SgEEzwxA57d
emyPxgcYxn/eR44/KJ4EBs+lVDR3veyJm+kXQ99b21/+jh5Xos1AnX5iItreGCc=
-----END CERTIFICATE-----
)EOF";

WiFiClientSecure espClient; // Use WiFiClientSecure for TLS
PubSubClient client(espClient);

// =========================================================

// Define GPIO pins for servos
#define PIN_SERVO_PAN  2 // GPIO2 for Pan (Right-Left)
#define PIN_SERVO_TILT 3 // GPIO3 for Tilt (Up-Down)

Servo servoPan;
Servo servoTilt;

// Initial servo positions
int currentPanPos = 90;  // 0-180
int currentTiltPos = 90; // 0-180

void setup_wifi() {
  Serial.print("Connecting to Wi-Fi ");
  Serial.print(ssid);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    attempts++;
    if (attempts > 20) {
      Serial.println("\nWi-Fi Connection Failed! Check SSID/Password or range.");
      break;
    }
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nSuccessfully connected to Wi-Fi!");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nCould not connect. Please restart ESP32 or check configuration.");
  }
}

void callback(char* topic, byte* payload, unsigned int length) {
  Serial.print("Message arrived [");
  Serial.print(topic);
  Serial.print("] ");
  String message = "";
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }
  Serial.println(message);

  int angle = message.toInt();

  if (angle >= 0 && angle <= 180) {
    if (String(topic) == MQTT_TOPIC_PAN_CMD) {
      servoPan.write(angle);
      currentPanPos = angle;
      Serial.print("Pan set to: ");
      Serial.println(angle);
      // Publish status after movement, using retained message for initial state
      client.publish(MQTT_TOPIC_PAN_STATUS, String(currentPanPos).c_str(), true);
    } else if (String(topic) == MQTT_TOPIC_TILT_CMD) {
      servoTilt.write(angle);
      currentTiltPos = angle;
      Serial.print("Tilt set to: ");
      Serial.println(angle);
      // Publish status after movement, using retained message for initial state
      client.publish(MQTT_TOPIC_TILT_STATUS, String(currentTiltPos).c_str(), true);
    }
  } else {
    Serial.println("Received invalid angle.");
  }
}

void reconnect() {
  // Loop until we're reconnected
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    // Create a random client ID
    String clientId = "ESP32Client-";
    clientId += String(random(0xffff), HEX);
    // Attempt to connect with username and password
    if (client.connect(clientId.c_str(), mqtt_username, mqtt_password)) {
      Serial.println("connected");
      // Once connected, publish an announcement and subscribe to topics
      client.publish("esp32/connection_status", "ESP32 is online");
      client.subscribe(MQTT_TOPIC_PAN_CMD);
      client.subscribe(MQTT_TOPIC_TILT_CMD);
      Serial.print("Subscribed to "); Serial.println(MQTT_TOPIC_PAN_CMD);
      Serial.print("Subscribed to "); Serial.println(MQTT_TOPIC_TILT_CMD);

      // Publish initial positions with retain flag set to true
      // This ensures that new subscribers immediately get the last known state.
      client.publish(MQTT_TOPIC_PAN_STATUS, String(currentPanPos).c_str(), true);
      client.publish(MQTT_TOPIC_TILT_STATUS, String(currentTiltPos).c_str(), true);

    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" trying again in 5 seconds");
      // Wait 5 seconds before retrying
      delay(5000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  Serial.println("\n");
  Serial.println("ESP32 Pan-Tilt Control with MQTT");
  Serial.println("------------------------------------");

  servoPan.attach(PIN_SERVO_PAN);
  servoTilt.attach(PIN_SERVO_TILT);

  // Set initial positions
  servoPan.write(currentPanPos);
  servoTilt.write(currentTiltPos);

  setup_wifi();

  // Set the CA certificate for TLS connection
  // Ensure this certificate is correct for your HiveMQ Cloud instance.
  espClient.setCACert(hivemq_ca_cert);

  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);
}

void loop() {
  // If not connected to MQTT, try to reconnect
  if (!client.connected()) {
    reconnect();
  }
  // Process incoming MQTT messages and maintain connection
  client.loop();
}
