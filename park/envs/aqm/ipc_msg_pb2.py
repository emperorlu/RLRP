# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ipc_msg.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)




DESCRIPTOR = _descriptor.FileDescriptor(
  name='ipc_msg.proto',
  package='rl',
  serialized_pb='\n\ripc_msg.proto\x12\x02rl\"f\n\nIPCMessage\x12\x0b\n\x03msg\x18\x01 \x01(\t\x12\x0b\n\x03\x65qc\x18\x02 \x01(\x04\x12\x0b\n\x03\x64qc\x18\x03 \x01(\x04\x12\x0b\n\x03\x65qb\x18\x04 \x01(\x04\x12\x0e\n\x06qdelay\x18\x05 \x01(\x04\x12\x14\n\x0c\x63urrent_prob\x18\x06 \x01(\x02\"%\n\x08IPCReply\x12\x0b\n\x03msg\x18\x01 \x01(\t\x12\x0c\n\x04prob\x18\x02 \x01(\x02')




_IPCMESSAGE = _descriptor.Descriptor(
  name='IPCMessage',
  full_name='rl.IPCMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='msg', full_name='rl.IPCMessage.msg', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=unicode("", "utf-8"),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='eqc', full_name='rl.IPCMessage.eqc', index=1,
      number=2, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='dqc', full_name='rl.IPCMessage.dqc', index=2,
      number=3, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='eqb', full_name='rl.IPCMessage.eqb', index=3,
      number=4, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='qdelay', full_name='rl.IPCMessage.qdelay', index=4,
      number=5, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='current_prob', full_name='rl.IPCMessage.current_prob', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=21,
  serialized_end=123,
)


_IPCREPLY = _descriptor.Descriptor(
  name='IPCReply',
  full_name='rl.IPCReply',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='msg', full_name='rl.IPCReply.msg', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=unicode("", "utf-8"),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='prob', full_name='rl.IPCReply.prob', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=125,
  serialized_end=162,
)

DESCRIPTOR.message_types_by_name['IPCMessage'] = _IPCMESSAGE
DESCRIPTOR.message_types_by_name['IPCReply'] = _IPCREPLY

class IPCMessage(_message.Message):
  __metaclass__ = _reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _IPCMESSAGE

  # @@protoc_insertion_point(class_scope:rl.IPCMessage)

class IPCReply(_message.Message):
  __metaclass__ = _reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _IPCREPLY

  # @@protoc_insertion_point(class_scope:rl.IPCReply)


# @@protoc_insertion_point(module_scope)
