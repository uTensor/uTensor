/***************************************
 * Generated Test
 ***************************************/

TEST({{ test_group }}, {{ test_name }}) {
  localCircularArenaAllocator<1024> meta_allocator;
  localCircularArenaAllocator<{{ out_size }}*2*sizeof(float), uint32_t> ram_allocator;
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);

  {% for tensor_decl in tensor_declarations %}
  {{ tensor_decl }}{% endfor %}

  {{ op_decl }}
  {{ op_eval }}

  {% for compare_snippet in compare_snippets %}
  {{ compare_snippet }}
  {% endfor %}
}
