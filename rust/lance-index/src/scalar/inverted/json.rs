// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{Array, ArrayRef, LargeBinaryArray, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::execution::{RecordBatchStream, SendableRecordBatchStream};
use futures::Stream;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

/// Transform jsonb stream into json text stream
pub struct JsonTextStream {
    inner: SendableRecordBatchStream,
    jsonb_col: String,
    schema: SchemaRef,
}

impl JsonTextStream {
    pub fn new(inner: SendableRecordBatchStream, jsonb_col: String) -> Self {
        let schema = json_text_schema(inner.schema().as_ref(), &jsonb_col);
        Self {
            inner,
            jsonb_col,
            schema,
        }
    }
}

/// The schema `JsonTextStream` yields: every input field, with `jsonb_col`
/// retyped from JSONB to text.
///
/// This must mirror what `poll_next` builds. Reporting only `jsonb_col` would
/// make `schema()` disagree with the batches on both field count and position, so
/// a caller resolving another column by name would fail, and one resolving
/// `jsonb_col` positionally would silently read whichever column happens to sit
/// at position 0.
fn json_text_schema(input_schema: &Schema, jsonb_col: &str) -> SchemaRef {
    Arc::new(Schema::new(
        input_schema
            .fields()
            .iter()
            .map(|field| {
                if field.name().as_str() == jsonb_col {
                    Field::new(jsonb_col, DataType::LargeUtf8, true)
                } else {
                    field.as_ref().clone()
                }
            })
            .collect::<Vec<Field>>(),
    ))
}

impl Stream for JsonTextStream {
    type Item = datafusion_common::Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.inner).poll_next(cx) {
            Poll::Ready(Some(Ok(batch))) => {
                let cols: Vec<ArrayRef> = batch
                    .schema()
                    .fields()
                    .iter()
                    .enumerate()
                    .map(|(idx, col)| {
                        if col.name().as_str() == self.jsonb_col {
                            Ok(jsonb_to_json(batch.column(idx), &self.jsonb_col)?)
                        } else {
                            Ok(batch.column(idx).clone())
                        }
                    })
                    .collect::<lance_core::Result<Vec<ArrayRef>>>()?;

                let mapped = RecordBatch::try_new(self.schema.clone(), cols).unwrap();
                Poll::Ready(Some(Ok(mapped)))
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl RecordBatchStream for JsonTextStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

pub fn jsonb_to_json(col: &ArrayRef, col_name: &str) -> lance_core::Result<ArrayRef> {
    let binary_array = col
        .as_any()
        .downcast_ref::<LargeBinaryArray>()
        .unwrap_or_else(|| panic!("column {} is not a large binary array", col_name));
    let mut builder =
        arrow_array::builder::LargeStringBuilder::with_capacity(binary_array.len(), 1024);
    for i in 0..binary_array.len() {
        if binary_array.is_null(i) {
            builder.append_null();
        } else if let Some(bytes) = binary_array.value(i).into() {
            let raw_jsonb = jsonb::RawJsonb::new(bytes);
            let json_text = raw_jsonb.to_string();
            builder.append_value(json_text);
        } else {
            unreachable!("jsonb value is not valid");
        }
    }
    Ok(Arc::new(builder.finish()))
}

#[cfg(test)]
mod tests {
    use crate::scalar::inverted::json::JsonTextStream;
    use arrow_array::builder::{LargeBinaryBuilder, UInt64Builder};
    use arrow_array::cast::AsArray;
    use arrow_array::{ArrayRef, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use datafusion::execution::RecordBatchStream;
    use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
    use futures::{TryStreamExt, stream};
    use serde_json::Value;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_json_text_stream() {
        let json_strings = [
            r#"{"a": 1, "b": "hello"}"#,
            r#"{"c": [1, 2, 3], "d": {"e": true}}"#,
            r#"{"f": null}"#,
        ];

        let mut jsonb_builder = LargeBinaryBuilder::new();
        let mut rowid_builder = UInt64Builder::new();

        for (i, json_str) in json_strings.iter().enumerate() {
            let jsonb_bytes = jsonb::parse_value(json_str.as_bytes()).unwrap().to_vec();
            jsonb_builder.append_value(jsonb_bytes);
            rowid_builder.append_value(i as u64);
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("json_col", DataType::LargeBinary, true),
            Field::new("rowid", DataType::UInt64, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(jsonb_builder.finish()) as ArrayRef,
                Arc::new(rowid_builder.finish()) as ArrayRef,
            ],
        )
        .unwrap();

        let stream = Box::pin(RecordBatchStreamAdapter::new(
            schema.clone(),
            stream::once(async { Ok(batch) }),
        ));

        let json_text_stream = JsonTextStream::new(stream, "json_col".to_string());
        // The declared stream schema must match the batches. Callers resolve other
        // columns by name and this one by index against it, so a schema listing
        // only the JSONB column would break both.
        let declared_schema = json_text_stream.schema();

        let result_batches: Vec<RecordBatch> = json_text_stream.try_collect().await.unwrap();
        assert_eq!(result_batches.len(), 1);
        let result_batch = &result_batches[0];

        let expected_schema = Arc::new(Schema::new(vec![
            Field::new("json_col", DataType::LargeUtf8, true),
            Field::new("rowid", DataType::UInt64, false),
        ]));
        assert_eq!(declared_schema, expected_schema);
        assert_eq!(result_batch.schema(), expected_schema);

        let json_text_col = result_batch
            .column_by_name("json_col")
            .unwrap()
            .as_string::<i64>();

        for (i, original_json_str) in json_strings.iter().enumerate() {
            let converted_json_str = json_text_col.value(i);
            let original_value: Value = serde_json::from_str(original_json_str).unwrap();
            let converted_value: Value = serde_json::from_str(converted_json_str).unwrap();
            assert_eq!(original_value, converted_value);
        }
    }
}
