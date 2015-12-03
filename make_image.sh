cell=$1

dot cell_${cell}.dot -Tpdf -o cell.pdf
evince cell.pdf &
