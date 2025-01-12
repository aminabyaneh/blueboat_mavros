##########################################################################################
# This test script reads the GLOBAL_POSITION_INT (GPS) message from the MAVLink connection
# and prints it to the console. UDP-connection pre-configured to work with BlueBoat.

# Important: Requires setup of secondary MAVLink connection in BlueBoat using port 14770.
##########################################################################################

from pymavlink import mavutil

def main():
      # Start a connection listening to a UDP port
      connection = mavutil.mavlink_connection('udpin:192.168.2.1:14770')

      # Wait for the first heartbeat
      # This sets the system and component ID of remote system for the link
      connection.wait_heartbeat()
      print(f'Heartbeat from the system (system {connection.target_system} component {connection.target_component})')

      while True:
            msg = connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
            if msg:
                  print(msg)

if __name__ == "__main__":
      main()