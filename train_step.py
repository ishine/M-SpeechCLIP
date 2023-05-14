import torch as th

def train_step(args, batch, model, loss_op, lang_to_skip=None):
    """
    Performs a forward pass and computes loss
    Doesn't call loss.backward or opt.step

    Allows multiple training steps in multilingual training w/ monolingual batches, but only one training step in all other cases
    """
    
    # Apply model to batch
    if args.model_type == 'Parallel' and args.loss_type != 'CrossLingual':
        image_out, text_out = model(batch['image'].to(args.device), batch['caption'].to(args.device))
    elif args.model_type == 'Parallel':
        # Cross-lingual losses
        if args.use_all_three:
            image_out, eng_out = model(batch['image'].to(args.device), batch['eng_caption'].to(args.device), langID=0)
            _, jpn_out = model(batch['image'].to(args.device), batch['jpn_caption'].to(args.device), langID=1)
            _, hindi_out = model(batch['image'].to(args.device), batch['hindi_caption'].to(args.device), langID=2)
        else:
            # Sampling two languages to compare
            if lang_to_skip == 'Eng':
                image_out, jpn_out = model(batch['image'].to(args.device), batch['jpn_caption'].to(args.device), langID=1)
                hindi_out = model(None, batch['hindi_caption'].to(args.device), langID=2)
            elif lang_to_skip == 'Jpn':
                image_out, eng_out = model(batch['image'].to(args.device), batch['eng_caption'].to(args.device), langID=0)
                hindi_out = model(None, batch['hindi_caption'].to(args.device), langID=2)
            elif lang_to_skip == 'Hindi':
                image_out, eng_out = model(batch['image'].to(args.device), batch['eng_caption'].to(args.device), langID=0)
                jpn_out = model(None, batch['jpn_caption'].to(args.device), langID=1)
    elif args.model_type == 'LangID': # Assuming LangID will never be used with CrossLingual loss
        image_out, text_out = model(batch['image'].to(args.device), batch['caption'].to(args.device), langID=batch['langID'].to(args.device))
    
    # Compute loss based on batch
    if args.loss_type == 'CrossLingual':
        if args.use_all_three:
            sim_ei = th.matmul(image_out.half(), eng_out.half().t())
            sim_ji = th.matmul(image_out.half(), jpn_out.half().t())
            sim_hi = th.matmul(image_out.half(), hindi_out.half().t())
            sim_ej = th.matmul(eng_out.half(), jpn_out.half().t())
            sim_eh = th.matmul(eng_out.half(), hindi_out.half().t())
            sim_jh = th.matmul(jpn_out.half(), hindi_out.half().t())
            
            mms_ei = loss_op(sim_ei)
            mms_ji = loss_op(sim_ji)
            mms_hi = loss_op(sim_hi)
            mms_ej = loss_op(sim_ej)
            mms_eh = loss_op(sim_eh)
            mms_jh = loss_op(sim_jh)

            total_loss = (mms_ei + mms_ji + mms_hi) + (mms_ej + mms_eh + mms_jh)*(args.cross_scale)
            return total_loss, mms_ei, mms_ji, mms_hi, mms_ej, mms_eh, mms_jh
        elif lang_to_skip == 'Eng':
            sim_ji = th.matmul(image_out.half(), jpn_out.half().t())
            sim_hi = th.matmul(image_out.half(), hindi_out.half().t())
            sim_jh = th.matmul(jpn_out.half(), hindi_out.half().t())
            
            mms_ji = loss_op(sim_ji)
            mms_hi = loss_op(sim_hi)
            mms_jh = loss_op(sim_jh)

            total_loss = (mms_ji + mms_hi) + mms_jh*args.cross_scale
            return total_loss, mms_ji, mms_hi, mms_jh
        elif lang_to_skip == 'Jpn':
            sim_ei = th.matmul(image_out.half(), eng_out.half().t())
            sim_hi = th.matmul(image_out.half(), hindi_out.half().t())
            sim_eh = th.matmul(eng_out.half(), hindi_out.half().t())
            
            mms_ei = loss_op(sim_ei)
            mms_hi = loss_op(sim_hi)
            mms_eh = loss_op(sim_eh)
                
            total_loss = (mms_ei + mms_hi) + mms_eh*args.cross_scale
            return total_loss, mms_ei, mms_hi, mms_eh
        elif lang_to_skip == 'Hindi':
            sim_ji = th.matmul(image_out.half(), jpn_out.half().t())
            sim_ei = th.matmul(image_out.half(), eng_out.half().t())
            sim_ej = th.matmul(eng_out.half(), jpn_out.half().t())
            
            mms_ji = loss_op(sim_ji)
            mms_ei = loss_op(sim_ei)
            mms_ej = loss_op(sim_ej)

            total_loss = (mms_ji + mms_ei) + mms_ej*args.cross_scale
            return total_loss, mms_ei, mms_ji, mms_ej 
    elif args.loss_type == 'MMS':
        sim_out = th.matmul(image_out.half(), text_out.half().t())
        total_loss = loss_op(sim_out)
        return total_loss
