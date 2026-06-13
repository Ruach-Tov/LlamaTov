%% SPDX-License-Identifier: GPL-2.0-or-later OR LicenseRef-RTAAL-1.1
%% Copyright (c) 2026 Heath Hunnicutt and the Ruach Tov collective.
% BPD Prolog IR — generated from file-formats/gguf/gguf.bpd
% 46 clauses, 0 directives

:- discontiguous state/2.
:- discontiguous initial_state/2.
:- discontiguous final_state/2.
:- discontiguous conditional/2.
:- discontiguous transition/4.
:- discontiguous transition_conditional/3.
:- discontiguous on/3.
:- discontiguous state_machine_template/2.
:- discontiguous parameter/4.
:- discontiguous variable/4.
:- discontiguous constant/3.
:- discontiguous dimension/3.
:- discontiguous conformance/2.
:- discontiguous invariant/2.
:- discontiguous process/1.
:- discontiguous actor/1.
:- discontiguous boundary/3.
:- discontiguous transport/2.
:- discontiguous framing/2.
:- discontiguous direction/2.
:- discontiguous absorbs/2.
:- discontiguous lifetime/3.
:- discontiguous derives_from/2.
:- discontiguous bpd_format/1.
:- discontiguous format_section/3.
:- discontiguous section/2.
:- discontiguous section_size/2.
:- discontiguous field/5.
:- discontiguous repeated/3.
:- discontiguous repeated/4.
:- discontiguous repeated_until/3.
:- discontiguous section_content/2.
:- discontiguous align_to/2.
:- discontiguous dispatch/4.
bpd_format(gguf).

format_endianness(gguf, little).

format_section(gguf, header, order(0)).

section(header, fixed_width).

section_size(header, byte_count(24)).

field(header, magic, byte_offset(0), magic_constant(bytes(4)), [must_equal(byte_sequence(71, 71, 85, 70))]).

field(header, version, byte_offset(4), format_version(cardinal(u32)), [must_equal(3)]).

field(header, tensor_count, byte_offset(8), tensor_count(count(cardinal(u64))), [must_be_lt(100000)]).

field(header, metadata_kv_count, byte_offset(16), metadata_kv_count(count(cardinal(u64))), [must_be_lt(100000)]).

format_section(gguf, metadata_kv_section, order(1)).

section(metadata_kv_section, variable_width).

repeated(metadata_kv_section, item(metadata_kv_pair), count_from(field(header, metadata_kv_count))).

section(metadata_kv_pair, variable_width).

field(metadata_kv_pair, key, byte_offset(0), metadata_key(length_prefixed(u64, u8)), []).

field(metadata_kv_pair, value_type, byte_offset(after(key)), metadata_value_type(cardinal(u32)), [must_be_in([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])]).

dispatch(metadata_kv_pair, value, discriminated_by(value_type), [case(0, field(value, cardinal(u8), byte_count(1))), case(1, field(value, cardinal(i8), byte_count(1))), case(2, field(value, cardinal(u16), byte_count(2))), case(3, field(value, cardinal(i16), byte_count(2))), case(4, field(value, cardinal(u32), byte_count(4))), case(5, field(value, cardinal(i32), byte_count(4))), case(6, field(value, float(f32), byte_count(4))), case(7, field(value, boolean(u8), byte_count(1))), case(8, field(value, metadata_string(length_prefixed(u64, u8)))), case(9, sub_record(value, metadata_array)), case(10, field(value, cardinal(u64), byte_count(8))), case(11, field(value, cardinal(i64), byte_count(8))), case(12, field(value, float(f64), byte_count(8)))]).

section(metadata_array, variable_width).

field(metadata_array, element_type, byte_offset(0), metadata_value_type(cardinal(u32)), [must_be_in([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])]).

field(metadata_array, element_count, byte_offset(after(element_type)), array_element_count(count(cardinal(u64))), [must_be_lt(10000000)]).

repeated(metadata_array, item(array_element), count_from(field(metadata_array, element_count))).

dispatch(array_element, value, discriminated_by(field(metadata_array, element_type)), [case(0, field(value, cardinal(u8), byte_count(1))), case(1, field(value, cardinal(i8), byte_count(1))), case(2, field(value, cardinal(u16), byte_count(2))), case(3, field(value, cardinal(i16), byte_count(2))), case(4, field(value, cardinal(u32), byte_count(4))), case(5, field(value, cardinal(i32), byte_count(4))), case(6, field(value, float(f32), byte_count(4))), case(7, field(value, boolean(u8), byte_count(1))), case(8, field(value, metadata_string(length_prefixed(u64, u8)))), case(9, sub_record(value, metadata_array)), case(10, field(value, cardinal(u64), byte_count(8))), case(11, field(value, cardinal(i64), byte_count(8))), case(12, field(value, float(f64), byte_count(8)))]).

dimension(magic_constant, parent_substrate(bytes), description('a fixed byte sequence used to identify a file format')).

dimension(format_version, parent_dimension(cardinal), description('an ordinal version number for a file format')).

dimension(tensor_count, parent_dimension(count), description('count of tensor records in a GGUF file')).

dimension(metadata_kv_count, parent_dimension(count), description('count of metadata key-value records in a GGUF file')).

dimension(metadata_key, parent_substrate(length_prefixed), description('a UTF-8 string serving as a metadata key; layout is length-prefixed (u64 length + bytes)')).

dimension(metadata_value_type, parent_dimension(cardinal), description('uint32 discriminator selecting one of 13 GGUF metadata value types: u8, i8, u16, i16, u32, i32, f32, bool, string, array, u64, i64, f64')).

dimension(metadata_string, parent_substrate(length_prefixed), description('a UTF-8 string serving as a metadata value; layout is length-prefixed (u64 length + bytes)')).

dimension(array_element_count, parent_dimension(count), description('count of elements in a GGUF metadata array value')).

format_section(gguf, tensor_info_section, order(2)).

section(tensor_info_section, variable_width).

repeated(tensor_info_section, item(tensor_info_record), count_from(field(header, tensor_count))).

section(tensor_info_record, variable_width).

field(tensor_info_record, name, byte_offset(0), tensor_name(length_prefixed(u64, u8)), []).

field(tensor_info_record, n_dimensions, byte_offset(after(name)), tensor_ndim(cardinal(u32)), [must_be_lt(8)]).

repeated(tensor_info_record, name(dimensions), item(field(dimension_value, cardinal(u64), byte_count(8))), count_from(field(tensor_info_record, n_dimensions))).

field(tensor_info_record, type, byte_offset(after(dimensions)), tensor_type(cardinal(u32)), []).

field(tensor_info_record, offset, byte_offset(after(type)), tensor_data_offset(relative_offset(cardinal(u64), relative_to(tensor_data_section))), []).

format_section(gguf, tensor_data_section, order(3)).

section(tensor_data_section, variable_width).

align_to(tensor_data_section, byte_alignment(32)).

section_content(tensor_data_section, opaque_blob).

dimension(tensor_name, parent_substrate(length_prefixed), description('UTF-8 tensor name, length-prefixed')).

dimension(tensor_ndim, parent_dimension(cardinal), description('number of dimensions in a tensor shape')).

dimension(tensor_type, parent_dimension(cardinal), description('uint32 discriminator for tensor element/quantization type')).

dimension(tensor_data_offset, parent_dimension(relative_offset), description('byte offset of tensor data relative to tensor_data_section start')).


% ─── P-1 derived facts ───────────────────────────────
% First Principle: do not dereference bytes not backed
% by file bytes. See design/precondition-derivation.md.
% These facts are DERIVED from section/2, section_size/2,
% and field/5 — not written by the format author.

:- discontiguous requires/2.
:- discontiguous provides/2.

% Sections: length-check provides length_verified.
provides(check_section_length(header), length_verified(header)).

% Field extraction: requires length_verified, provides field_extracted.
requires(extract_field(header, magic), length_verified(header)).
provides(extract_field(header, magic), field_extracted(header, magic)).
requires(extract_field(header, metadata_kv_count), length_verified(header)).
provides(extract_field(header, metadata_kv_count), field_extracted(header, metadata_kv_count)).
requires(extract_field(header, tensor_count), length_verified(header)).
provides(extract_field(header, tensor_count), field_extracted(header, tensor_count)).
requires(extract_field(header, version), length_verified(header)).
provides(extract_field(header, version), field_extracted(header, version)).
provides(extract_field(metadata_array, element_count), field_extracted(metadata_array, element_count)).
provides(extract_field(metadata_array, element_type), field_extracted(metadata_array, element_type)).
provides(extract_field(metadata_kv_pair, key), field_extracted(metadata_kv_pair, key)).
provides(extract_field(metadata_kv_pair, value_type), field_extracted(metadata_kv_pair, value_type)).
provides(extract_field(tensor_info_record, n_dimensions), field_extracted(tensor_info_record, n_dimensions)).
provides(extract_field(tensor_info_record, name), field_extracted(tensor_info_record, name)).
provides(extract_field(tensor_info_record, offset), field_extracted(tensor_info_record, offset)).
provides(extract_field(tensor_info_record, type), field_extracted(tensor_info_record, type)).

% Field validation: requires field_extracted, provides field_validated.
requires(validate_field(header, magic), field_extracted(header, magic)).
provides(validate_field(header, magic), field_validated(header, magic)).
requires(validate_field(header, metadata_kv_count), field_extracted(header, metadata_kv_count)).
provides(validate_field(header, metadata_kv_count), field_validated(header, metadata_kv_count)).
requires(validate_field(header, tensor_count), field_extracted(header, tensor_count)).
provides(validate_field(header, tensor_count), field_validated(header, tensor_count)).
requires(validate_field(header, version), field_extracted(header, version)).
provides(validate_field(header, version), field_validated(header, version)).
requires(validate_field(metadata_array, element_count), field_extracted(metadata_array, element_count)).
provides(validate_field(metadata_array, element_count), field_validated(metadata_array, element_count)).
requires(validate_field(metadata_array, element_type), field_extracted(metadata_array, element_type)).
provides(validate_field(metadata_array, element_type), field_validated(metadata_array, element_type)).
requires(validate_field(metadata_kv_pair, value_type), field_extracted(metadata_kv_pair, value_type)).
provides(validate_field(metadata_kv_pair, value_type), field_validated(metadata_kv_pair, value_type)).
requires(validate_field(tensor_info_record, n_dimensions), field_extracted(tensor_info_record, n_dimensions)).
provides(validate_field(tensor_info_record, n_dimensions), field_validated(tensor_info_record, n_dimensions)).


% ─── Derived field byte sizes ──────────────────
% Computed from field/5 type terms via
% level0_substrate.size_in_bytes. Queried by
% section_layout/2 (ir/include/section_layout.pl) and
% by future code emitters.

:- discontiguous field_byte_size/3.

field_byte_size(header, magic, 4).
field_byte_size(header, version, 4).
field_byte_size(header, tensor_count, 8).
field_byte_size(header, metadata_kv_count, 8).
field_byte_size(metadata_kv_pair, value_type, 4).
field_byte_size(metadata_array, element_type, 4).
field_byte_size(metadata_array, element_count, 8).
field_byte_size(tensor_info_record, n_dimensions, 4).
field_byte_size(tensor_info_record, type, 4).
field_byte_size(tensor_info_record, offset, 8).