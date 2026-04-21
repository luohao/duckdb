use std::fs::File;
use parquet::file::reader::{FileReader, SerializedFileReader};

fn main() {
    for p in ["/tmp/kway_4t/k2_o050/t0.parquet"] {
        let file = File::open(p).unwrap();
        let reader = SerializedFileReader::new(file).unwrap();
        let meta = reader.metadata();
        println!(
            "{}: row_groups={}, total_rows={}",
            p,
            meta.num_row_groups(),
            meta.file_metadata().num_rows()
        );
        for i in 0..meta.num_row_groups() {
            let rg = meta.row_group(i);
            println!("  rg{}: {} rows", i, rg.num_rows());
        }
    }
}
