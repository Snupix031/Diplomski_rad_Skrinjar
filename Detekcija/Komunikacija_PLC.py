import snap7
from snap7.util import *
from snap7.type import *

# IP adresa PLC-a
PLC_IP = '192.168.1.3'  # prilagodi tvojoj mreži
RACK = 0
SLOT = 1  # za S7-1200 ili S7-1500, obično je SLOT 1

client = snap7.client.Client()
client.connect(PLC_IP, RACK, SLOT)

if client.get_connected():
    print("[INFO] Povezan s PLC-om ✅")
else:
    print("[ERROR] Neuspješno spajanje s PLC-om ❌")


# Pretvaranje float vrijednosti u 4 bajta (REAL za Siemens)
# def write_real_to_db(db_number, start_offset, value):
  #  data = bytearray(4)
   # struct.pack_into('>f', data, 0, value)  # Siemens koristi Big-Endian
#    client.db_write(db_number, start_offset, data)

# Primjer: upiši poziciju objekta 250.5 mm u DB1.DBD0
# write_real_to_db(1, 0, 250.5)