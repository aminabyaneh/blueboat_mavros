# Blue Boat MavROS Guide

Part A is focused on ROS networking setups and access to MavROS Topics. In part B, you can directly set up a UDP connection to the boat, and use MavLink Python package to communicate (still testing).

Code partly adopted from [this repository](https://github.com/ImStian/blueboat_globalpathplanner/tree/main).

## A. ROS Networking Setup

### **Step 0: Connect to the Blue Boat** 🌐

Before setting up the ROS network, ensure you are connected to the Blue Boat's WiFi router.

1. Connect to the WiFi Router:
    - Follow the instructions provided [on the software guide](https://bluerobotics.com/learn/blueboat-software-setup/) to connect your laptop to the Blue Boat's WiFi network.

2. Access BlueOS:
    - Make sure the Blue Boat is up and running based on the [setup guide](https://bluerobotics.com/learn/blueboat-general-integration-guide/), and ensure you can see it connected to the router based on the [software guide](https://bluerobotics.com/learn/blueboat-software-setup/).
    - Open a web browser and navigate to the BlueOS interface using the IP address provided [here](https://bluerobotics.com/learn/blueboat-software-setup/#verifying-the-network-connection).

3. Open the BlueOS ROS Interface:
    - Once in the BlueOS interface, navigate to the ROS section to access the ROS settings and tools.

> **⚠️ Important:** Ensure MavROS is up and running! You should not see any errors in the ROS panel of the BlueOS. If you do, it's most likely because the TCP port for MavROS is set wrong. Follow these steps to fix this problem:
>
> 1. Activate the pirate mode in BlueOS on the top right of the homepage.
> 2. Set the MavROS port to the same port as the TCP_USER port.
> 3. Run the MavROS command in the BlueOS ROS terminals again, using the correct TCP port.

---

### **Step 1: Find IP Addresses** 🔍

You need to determine the **IP addresses** of both the **robot** and the **laptop**.

#### **On the Robot Computer** 🤖

1. Run the following to get the IP:

    ```bash
    ip a | grep inet
    ```

    or

    ```bash
    hostname -I
    ```

2. Identify the local IP (e.g., `BLUE_OS_IP`). Avoid `127.0.0.1`, as it is a loopback address.
3. You might see two IPs for the BlueOS, pick the one you used to access BlueOS.

#### **On the Laptop** 💻

1. Run the same command:

    ```bash
    ip a | grep inet  # Linux/macOS
    ```

2. Identify the laptop's IP (e.g., `COMPUTER_IP`).

---

### **Step 2: Set Environment Variables** 🌍

Now, set the **ROS_MASTER_URI** and **ROS_IP** on both devices.

#### **On the Robot Computer** 🤖

1. Add the following commands (replace `ROBOT_IP` with the actual IP of the robot, e.g., `BLUE_OS_IP`):

    ```bash
    export ROS_MASTER_URI=http://192.168.2.2:11311
    export ROS_IP=192.168.2.2
    ```

2. To make these settings permanent, add the lines to the `~/.bashrc` file:

    ```bash
    echo "export ROS_MASTER_URI=http://BLUE_OS_IP:11311" >> ~/.bashrc
    echo "export ROS_IP=BLUE_OS_IP" >> ~/.bashrc
    source ~/.bashrc
    ```

> **⚠️ Critical:** This must be done for all terminals in the BlueBoat!! We are working on a solution to run this at boot-up on BlueOS.

#### **On the Laptop** 💻

1. Open a terminal.
2. Set the variables (replace `ROBOT_IP` and `LAPTOP_IP` with the actual values):

    ```bash
    export ROS_MASTER_URI=http://192.168.2.2:11311
    export ROS_IP=192.168.2.1
    ```

3. To make these permanent, add them to `~/.bashrc`:

    ```bash
    echo "export ROS_MASTER_URI=http://BLUE_OS_IP:11311" >> ~/.bashrc
    echo "export ROS_IP=COMPUTER_IP" >> ~/.bashrc
    source ~/.bashrc
    ```

---

### **Step 3: Test the Connection** 🔗

#### **On the Robot Computer** 🤖

* Check that `roscore` is active in BlueOS ROS terminals.

#### **On the Laptop** 💻

* Get a list of topics:

    ```bash
    rostopic list
    ```

    If everything is set up correctly, you should see the list of available `/mavros/*` topics.

---

### **Step 4: Troubleshooting (hopefully not)** 🛠️

If the connection is not working:

1. **Check IP addresses** on both computers:

    ```bash
    echo $ROS_IP
    echo $ROS_MASTER_URI
    ```

2. **Ping the robot from the laptop**:

    ```bash
    ping BLUE_OS_IP
    ```

    If there’s no response, ensure both are on the same network.
3. **Disable firewalls** temporarily to test:

    ```bash
    sudo ufw disable  # Linux
    ```

---

### MavROS Documentation 📚

For detailed information and advanced usage of MavROS, refer to the official [MavROS documentation](https://wiki.ros.org/mavros).

MavROS is a ROS package that provides communication between ROS and MAVLink-based autopilots. It includes various plugins that allow you to interact with different MAVLink messages and services.

> 📝 **Note**: By following the MavROS documentation, you can leverage the full capabilities of your MAVLink-compatible Blue Boat within the ROS ecosystem.

### Designated Topics (Not needed when using MavLink Python package!)

These are the topics we found useful during the initial testing phase. Feel free to add to the list.

| Topic                       | Message Type             | Description                                      |
|-----------------------------|--------------------------|--------------------------------------------------|
| /mavros/state               | mavros_msgs/State        | Current state of the MAVLink device              |
| /mavros/imu/data            | sensor_msgs/Imu          | IMU data including orientation and angular velocity|
| /mavros/local_position/pose | geometry_msgs/PoseStamped| Local position of the MAVLink device             |
| /mavros/rc/out              | mavros_msgs/RCOut        | RC output values                                 |
| /mavros/battery             | sensor_msgs/BatteryState | Battery status of the MAVLink device             |

### Designated Services

These are the services we found useful during the initial testing phase. Feel free to add to the list.

| Service                  | Service Type               | Description                                      |
|--------------------------|----------------------------|--------------------------------------------------|
| /mavros/cmd/arming       | mavros_msgs/CommandBool    | Arm or disarm the MAVLink device                 |
| /mavros/set_mode         | mavros_msgs/SetMode        | Set the flight mode of the MAVLink device        |
| /mavros/param/get        | mavros_msgs/ParamGet       | Retrieve a parameter from the MAVLink device     |
| /mavros/param/set        | mavros_msgs/ParamSet       | Set a parameter on the MAVLink device            |
| /mavros/command/takeoff  | mavros_msgs/CommandTOL     | Command the MAVLink device to take off           |
| /mavros/command/land     | mavros_msgs/CommandTOL     | Command the MAVLink device to land               |


## MAVLink Communication Module

Python MavLink package creates a reliable UDP link with the boat for planning missions, and basically all functionalities of QGroundControl.


### Link Test

Run the script [`mavlink_udp_heartbeat.py`](mavlink/mavlink_udp_heartbeat.py). You need to have the boat and base station set up, and your host machine connected to the same base station network.

You can use the BlueOS webpage, typically on `192.168.2.2` in the local network, to check Mavlink server address and ports.

> **⚠️ Note:** We successfully tested this using the **GCS_Client_Link** port and IP. This was set to `192.168.2.2:14550` at the time. It can be found in the **MavLink Endpoints** menu, in **pirate mode**.


### MavLink Python Library

Use the functions provided in [`mavlink_library.py`](mavlink/mavlink_library.py) to build scripts like [`mavlink_data_logger.py`](mavlink/mavlink_data_logger.py). These scripts rely on a reliable connection to Mavlink servers on the boat through the base station.
