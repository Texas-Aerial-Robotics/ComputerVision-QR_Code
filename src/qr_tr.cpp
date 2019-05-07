#include <functional>

#include "qr.h"

/* 
 * Don't know anything about the masks yet, so just hold onto the 
 * stuff we grab from this tr corner until we get the info from the br
 */

uint16_t compWithTR(int width, qr::qr_t top) {

}

uint16_t qr::internal::compute_tr(qr::computed_qr_t top){
  return compWithTR(top.code().width(), top.code());
}
