for(int i = 0; i < {{ len(b.flatten()) }}; i++) {
  EXPECT_NEAR(static_cast<{{ a.dtype }}>( {{ a.name }}(i) ), static_cast<{{ b.dtype }}>( {{ b.name }}(i) ), {{ threshold }});
}
